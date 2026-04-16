param(
    [string]$TunnelHost,
    [int]$TunnelPort,

    [string]$Alias = "colab-gpu",
    [string]$User = "colab",
    [string]$IdentityFile = "$HOME\.ssh\id_ed25519",
    [switch]$PrintPublicKeyOnly,
    [switch]$ConnectNow,
    [switch]$OpenVscode
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-SshKeygenPath {
    $cmd = Get-Command ssh-keygen -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $fallback = "C:\Windows\System32\OpenSSH\ssh-keygen.exe"
    if (Test-Path $fallback) { return $fallback }

    throw "ssh-keygen not found. Install OpenSSH Client and retry."
}

function Ensure-SshKey {
    param(
        [string]$KeyPath
    )

    $pubPath = "$KeyPath.pub"
    if (-not (Test-Path $KeyPath) -or -not (Test-Path $pubPath)) {
        Write-Host "SSH key not found. Creating key: $KeyPath"
        # Attempt 1: native invocation.
        $sshKeygenArgs = @(
            "-t", "ed25519",
            "-f", $KeyPath,
            "-N", ""
        )
        $sshKeygenPath = Resolve-SshKeygenPath
        & $sshKeygenPath @sshKeygenArgs | Out-Null
        $nativeExit = $LASTEXITCODE

        # Attempt 2: fallback for Windows PowerShell empty-arg bug with native commands.
        if ($nativeExit -ne 0) {
            Write-Host "Native ssh-keygen call failed (exit=$nativeExit). Retrying via cmd.exe fallback..."
            $cmdLine = "`"$sshKeygenPath`" -t ed25519 -f `"$KeyPath`" -N `"`""
            & cmd.exe /c $cmdLine | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "ssh-keygen failed with exit code $LASTEXITCODE"
            }
        }
    }

    if (-not (Test-Path $KeyPath) -or -not (Test-Path $pubPath)) {
        throw "SSH key files were not created: $KeyPath and/or $pubPath"
    }
}

function Ensure-SshConfigEntry {
    param(
        [string]$ConfigPath,
        [string]$HostAlias,
        [string]$HostName,
        [int]$Port,
        [string]$RemoteUser,
        [string]$KeyPath
    )

    $entry = @"
Host $HostAlias
    HostName $HostName
    Port $Port
    User $RemoteUser
    IdentityFile $KeyPath
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
    StrictHostKeyChecking no
    UserKnownHostsFile NUL

"@

    $existing = ""
    if (Test-Path $ConfigPath) {
        $existing = Get-Content $ConfigPath -Raw
    } else {
        New-Item -ItemType File -Path $ConfigPath -Force | Out-Null
    }

    # Replace existing block for alias if present.
    $pattern = "(?ms)^Host\s+$([Regex]::Escape($HostAlias))\s.*?(?=^Host\s|\z)"
    if ($existing -match $pattern) {
        $updated = [Regex]::Replace($existing, $pattern, $entry)
    } else {
        $updated = $existing.TrimEnd() + "`r`n`r`n" + $entry
    }
    Set-Content -Path $ConfigPath -Value $updated -Encoding UTF8
}

$sshDir = Join-Path $HOME ".ssh"
if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
}

Ensure-SshKey -KeyPath $IdentityFile
$pubPath = "$IdentityFile.pub"
$pubKey = Get-Content $pubPath -Raw

if ($PrintPublicKeyOnly) {
    Write-Host $pubKey
    exit 0
}

if ([string]::IsNullOrWhiteSpace($TunnelHost) -or $TunnelPort -le 0) {
    throw "TunnelHost and TunnelPort are required unless -PrintPublicKeyOnly is used."
}

$configPath = Join-Path $sshDir "config"
Ensure-SshConfigEntry `
    -ConfigPath $configPath `
    -HostAlias $Alias `
    -HostName $TunnelHost `
    -Port $TunnelPort `
    -RemoteUser $User `
    -KeyPath $IdentityFile

Write-Host ""
Write-Host "Done."
Write-Host "Alias: $Alias"
Write-Host "SSH config: $configPath"
Write-Host "Public key (copy to Colab AUTHORIZED_KEY):"
Write-Host $pubKey
Write-Host ""
Write-Host "Test command:"
Write-Host "ssh $Alias"
Write-Host ""
Write-Host "VS Code command:"
Write-Host "code --remote ssh-remote+$Alias /content"

if ($ConnectNow) {
    ssh $Alias
}

if ($OpenVscode) {
    code --remote "ssh-remote+$Alias" /content
}
