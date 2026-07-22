import Foundation
import FoundationModels
import Network

private struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data
}

private struct HTTPResponse {
    let status: Int
    let contentType: String
    let body: Data

    static func json(_ value: Any, status: Int = 200) -> HTTPResponse {
        let body = (try? JSONSerialization.data(withJSONObject: value)) ?? Data("{}".utf8)
        return HTTPResponse(status: status, contentType: "application/json", body: body)
    }
}

private final class AppleModelServer: @unchecked Sendable {
    private let listener: NWListener
    private let apiKey: String
    private let queue = DispatchQueue(label: "ai.mirid.apple-model")

    init(host: String, port: UInt16, apiKey: String) throws {
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw NSError(domain: "MiridAppleRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid port"])
        }
        self.listener = try NWListener(using: .tcp, on: nwPort)
        self.apiKey = apiKey
        if host != "127.0.0.1" && host != "localhost" {
            throw NSError(domain: "MiridAppleRunner", code: 2, userInfo: [NSLocalizedDescriptionKey: "Only loopback connections are allowed"])
        }
    }

    func start() {
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.stateUpdateHandler = { state in
            if case .failed(let error) = state {
                FileHandle.standardError.write(Data("Apple model server failed: \(error)\n".utf8))
                exit(1)
            }
        }
        listener.start(queue: queue)
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(connection, buffer: Data())
    }

    private func receive(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8 * 1024 * 1024) { [weak self] data, _, _, error in
            guard let self else { return }
            var next = buffer
            if let data {
                next.append(data)
            }
            if let request = self.parseRequest(next) {
                Task {
                    let response = await self.route(request)
                    self.send(response, through: connection)
                }
            } else if error == nil {
                self.receive(connection, buffer: next)
            } else {
                connection.cancel()
            }
        }
    }

    private func parseRequest(_ data: Data) -> HTTPRequest? {
        let delimiter = Data("\r\n\r\n".utf8)
        guard let headerRange = data.range(of: delimiter),
              let headerText = String(data: data[..<headerRange.lowerBound], encoding: .utf8) else {
            return nil
        }
        let lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        let requestParts = requestLine.split(separator: " ")
        guard requestParts.count >= 2 else { return nil }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                headers[String(parts[0]).lowercased()] = String(parts[1]).trimmingCharacters(in: .whitespaces)
            }
        }
        let bodyStart = headerRange.upperBound
        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        guard data.count >= bodyStart + contentLength else { return nil }
        let body = data.subdata(in: bodyStart..<(bodyStart + contentLength))
        return HTTPRequest(
            method: String(requestParts[0]),
            path: String(requestParts[1]),
            headers: headers,
            body: body
        )
    }

    private func authorised(_ request: HTTPRequest) -> Bool {
        request.headers["authorization"] == "Bearer \(apiKey)"
    }

    private func route(_ request: HTTPRequest) async -> HTTPResponse {
        if request.path == "/health" {
            return .json(["status": "ok"])
        }
        guard authorised(request) else {
            return .json(["error": ["message": "Unauthorised"]], status: 401)
        }
        if request.method == "GET" && request.path == "/v1/models" {
            return .json(["object": "list", "data": [["id": "mirid/apple-intelligence", "object": "model"]]])
        }
        guard request.method == "POST",
              request.path == "/v1/completions" || request.path == "/v1/chat/completions",
              let payload = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
            return .json(["error": ["message": "Not found"]], status: 404)
        }
        do {
            let isChat = request.path.contains("chat")
            let prompt = isChat ? chatPrompt(payload["messages"] as? [[String: Any]] ?? []) : (payload["prompt"] as? String ?? "")
            let content = try await generate(prompt: prompt)
            let stream = payload["stream"] as? Bool ?? false
            if stream {
                let choice: [String: Any] = isChat
                    ? ["index": 0, "delta": ["content": content], "finish_reason": NSNull()]
                    : ["index": 0, "text": content, "finish_reason": NSNull()]
                let event = try JSONSerialization.data(withJSONObject: ["choices": [choice]])
                var body = Data("data: ".utf8)
                body.append(event)
                body.append(Data("\n\ndata: [DONE]\n\n".utf8))
                return HTTPResponse(status: 200, contentType: "text/event-stream", body: body)
            }
            let choice: [String: Any] = isChat
                ? ["index": 0, "message": ["role": "assistant", "content": content], "finish_reason": "stop"]
                : ["index": 0, "text": content, "finish_reason": "stop"]
            return .json([
                "id": UUID().uuidString,
                "object": isChat ? "chat.completion" : "text_completion",
                "model": "mirid/apple-intelligence",
                "choices": [choice],
            ])
        } catch {
            return .json(["error": ["message": error.localizedDescription]], status: 500)
        }
    }

    private func chatPrompt(_ messages: [[String: Any]]) -> String {
        messages.compactMap { message in
            guard let role = message["role"] as? String,
                  let content = message["content"] as? String else { return nil }
            return "\(role.uppercased()): \(content)"
        }.joined(separator: "\n\n") + "\n\nASSISTANT:"
    }

    @available(macOS 26.0, *)
    private func respond(prompt: String) async throws -> String {
        let model = SystemLanguageModel.default
        guard model.isAvailable else {
            throw NSError(domain: "MiridAppleRunner", code: 3, userInfo: [NSLocalizedDescriptionKey: "Apple Intelligence is not ready on this Mac"])
        }
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(to: prompt)
        return response.content
    }

    private func generate(prompt: String) async throws -> String {
        if #available(macOS 26.0, *) {
            return try await respond(prompt: prompt)
        }
        throw NSError(domain: "MiridAppleRunner", code: 4, userInfo: [NSLocalizedDescriptionKey: "This version of macOS does not include Apple Intelligence model access"])
    }

    private func send(_ response: HTTPResponse, through connection: NWConnection) {
        let reason = response.status == 200 ? "OK" : response.status == 401 ? "Unauthorised" : "Error"
        var data = Data("HTTP/1.1 \(response.status) \(reason)\r\n".utf8)
        data.append(Data("Content-Type: \(response.contentType)\r\n".utf8))
        data.append(Data("Content-Length: \(response.body.count)\r\n".utf8))
        data.append(Data("Connection: close\r\n\r\n".utf8))
        data.append(response.body)
        connection.send(content: data, completion: .contentProcessed { _ in connection.cancel() })
    }
}

private func argument(_ name: String, default fallback: String) -> String {
    guard let index = CommandLine.arguments.firstIndex(of: name), index + 1 < CommandLine.arguments.count else {
        return fallback
    }
    return CommandLine.arguments[index + 1]
}

private func probe() -> Int32 {
    if #available(macOS 26.0, *) {
        let model = SystemLanguageModel.default
        if model.isAvailable {
            print("available")
            return 0
        }
        print("unavailable: \(model.availability)")
        return 1
    }
    print("unavailable: macOS 26 or later is required")
    return 1
}

if CommandLine.arguments.contains("--probe") {
    exit(probe())
}

let host = argument("--host", default: "127.0.0.1")
let port = UInt16(argument("--port", default: "0")) ?? 0
let apiKey = argument("--api-key", default: "")
do {
    let server = try AppleModelServer(host: host, port: port, apiKey: apiKey)
    server.start()
    dispatchMain()
} catch {
    FileHandle.standardError.write(Data("Apple model server could not start: \(error.localizedDescription)\n".utf8))
    exit(1)
}
