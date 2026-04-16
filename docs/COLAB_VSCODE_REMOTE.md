# Google Colab GPU in VS Code (Automated)

## Files

- Notebook (run in Colab): `notebooks/colab_vscode_remote_gpu.ipynb`
- Local Windows setup script: `scripts/setup_colab_vscode_remote.ps1`

## Quick flow

1. Get local public key:

```powershell
.\scripts\setup_colab_vscode_remote.ps1 -PrintPublicKeyOnly
```

2. Open and run `notebooks/colab_vscode_remote_gpu.ipynb` in Colab.
3. Paste your local SSH public key into `AUTHORIZED_KEY`.
4. Run all cells; notebook prints tunnel `host:port`.
5. On local Windows PowerShell, run:

```powershell
.\scripts\setup_colab_vscode_remote.ps1 -TunnelHost <HOST> -TunnelPort <PORT> -Alias colab-gpu -User colab -OpenVscode
```

6. VS Code opens remote target (`Remote-SSH`) and you can work in `/content` using Colab GPU runtime.

## Notes

- Colab endpoint changes after runtime restart/disconnect.
- Rerun notebook + local script each time a new runtime starts.
- This setup is best-effort and may break if Colab runtime/network policies change.
