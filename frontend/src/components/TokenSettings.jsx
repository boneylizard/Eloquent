import React from 'react';
import { useApp } from '../contexts/AppContext';
import { Label } from './ui/label';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';

const TokenSettings = () => {
  const { settings, updateSettings } = useApp();
  
  // Function to handle max tokens change
  const handleMaxTokensChange = (value) => {
    const tokenValue = value === 'auto' ? -1 : parseInt(value, 10);
    updateSettings({ max_tokens: tokenValue });
  };

  // Function to handle context length change  
  const handleContextLengthChange = (value) => {
    updateSettings({ contextLength: parseInt(value, 10) });
  };
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Token Settings</CardTitle>
        <CardDescription>Configure model token limits and context window</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Max Tokens Setting */}
        <div className="space-y-2">
          <Label htmlFor="max-tokens">Maximum Response Length</Label>
          <Select 
            value={settings.max_tokens === -1 ? 'auto' : settings.max_tokens.toString()}
            onValueChange={handleMaxTokensChange}
          >
            <SelectTrigger id="max-tokens" className="w-full">
              <SelectValue placeholder="Select max tokens" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto (Maximum Available)</SelectItem>
              <SelectItem value="1024">1024 tokens (Short)</SelectItem>
              <SelectItem value="2048">2048 tokens (Medium)</SelectItem>
              <SelectItem value="4096">4096 tokens (Long)</SelectItem>
              <SelectItem value="8192">8192 tokens (Very Long)</SelectItem>
              <SelectItem value="16384">16384 tokens (Maximum)</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            'Auto' will use maximum available length based on model context window.
          </p>
        </div>

        {/* Context Length Setting */}
        <div className="space-y-2">
          <Label htmlFor="context-length">Model Context Length</Label>
          <Select
            value={settings.contextLength.toString()}
            onValueChange={handleContextLengthChange}
          >
            <SelectTrigger id="context-length" className="w-full">
              <SelectValue placeholder="Select context length" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="8192">8K</SelectItem>
              <SelectItem value="16384">16K</SelectItem>
              <SelectItem value="32768">32K</SelectItem>
              <SelectItem value="65536">64K</SelectItem>
              <SelectItem value="131072">128K</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Set the context length when loading models. Higher values allow longer conversations but use more memory.
          </p>
        </div>

        {/* Manual Input Option */}
        <div className="space-y-2">
          <Label htmlFor="custom-max-tokens">Custom Max Tokens</Label>
          <Input
            id="custom-max-tokens"
            type="number"
            min="-1"
            step="1"
            value={settings.max_tokens}
            onChange={(e) => updateSettings({ max_tokens: parseInt(e.target.value, 10) })}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            Use -1 for auto-sizing, or enter a specific value. Larger values allow longer responses.
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default TokenSettings;