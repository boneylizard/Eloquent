/**
 * Python Drive Mapper — maps validated Python drive descriptors
 * to frontend effect presets.
 *
 * The backend validates Python drives via AST (python_validator.py) and
 * sends them as descriptor objects: { validated, drives: [{ action, args }] }
 *
 * This module maps those descriptors to CSS effect parameters that
 * InterfaceHijackContext can apply. NO Python is ever executed.
 */

/**
 * Map validated Python drives to interface hijack effect parameters.
 *
 * @param {Object} pythonDrive - { validated, drives } or { validated, errors }
 * @returns {Object|null} - hijack effect descriptor, or null if no drives
 */
export function mapPythonDrives(pythonDrive) {
  if (!pythonDrive || !pythonDrive.validated || !pythonDrive.drives) {
    return null;
  }

  const result = {
    theme_shift: null,
    shake: { intensity: 0.0, duration_ms: 0 },
    lock: { input_locked: false, scroll_locked: false, duration_ms: 0 },
    glitch: { intensity: 0.0 },
  };

  for (const drive of pythonDrive.drives) {
    const { action, args } = drive;
    switch (action) {
      case 'grip':
        result.lock.input_locked = true;
        result.lock.duration_ms = Math.max(
          result.lock.duration_ms,
          Math.round((args.intensity || 0.5) * 5000)
        );
        break;
      case 'shock':
        result.shake.intensity = Math.max(result.shake.intensity, args.intensity || 0.5);
        result.shake.duration_ms = Math.max(
          result.shake.duration_ms,
          Math.round((args.duration || 0.5) * 3000)
        );
        result.glitch.intensity = Math.max(
          result.glitch.intensity,
          (args.intensity || 0.5) * 0.7
        );
        break;
      case 'freeze':
        result.lock.input_locked = true;
        result.lock.scroll_locked = true;
        result.lock.duration_ms = Math.max(
          result.lock.duration_ms,
          Math.round((args.duration || 0.5) * 5000)
        );
        break;
      case 'theme':
        result.theme_shift = _paletteToTheme(args.palette);
        break;
      case 'whisper':
        // Whisper is a ghost signal visual — no hijack effect
        break;
    }
  }

  return result;
}

const PALETTE_MAP = {
  crimson: { hue: 0, saturation: 1.4, brightness: 0.7 },
  ice: { hue: 180, saturation: 0.8, brightness: 1.1 },
  void: { hue: 270, saturation: 0.6, brightness: 0.4 },
  amber: { hue: 30, saturation: 1.3, brightness: 0.9 },
  ash: { hue: 0, saturation: 0.2, brightness: 0.5 },
  pulse: { hue: 120, saturation: 1.5, brightness: 1.0 },
  depths: { hue: 210, saturation: 1.0, brightness: 0.3 },
  standard: null, // no shift
};

function _paletteToTheme(palette) {
  return PALETTE_MAP[palette] || null;
}
