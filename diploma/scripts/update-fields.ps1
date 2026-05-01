# Pre-render docx fields via Word COM automation.
#
# Why: Pandoc emits a TOC field with w:dirty="true" so Word fills it on first
# open. We sweep dirty="true" -> "false" in postprocess to suppress the
# "update fields?" prompt, but that also leaves the TOC empty. This script
# runs between Pandoc and postprocess: it opens the docx in Word headlessly,
# updates TablesOfContents and all other fields (so the TOC content is baked
# into document.xml as static runs), saves, and exits. Postprocess afterwards
# resets dirty="false" on the now-populated fields.
#
# Requires Microsoft Word installed (uses COM Automation).
# Usage: powershell -File update-fields.ps1 <docx-path>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$DocxPath
)

$ErrorActionPreference = 'Stop'

$absPath = (Resolve-Path -LiteralPath $DocxPath).Path
Write-Host "[update-fields] opening $absPath"

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0  # wdAlertsNone

    # Open(FileName, ConfirmConversions=$false, ReadOnly=$false, AddToRecentFiles=$false)
    $doc = $word.Documents.Open($absPath, $false, $false, $false)

    $tocCount = $doc.TablesOfContents.Count
    for ($i = 1; $i -le $tocCount; $i++) {
        $doc.TablesOfContents.Item($i).Update()
    }

    $fieldCount = $doc.Fields.Count
    if ($fieldCount -gt 0) {
        $doc.Fields.Update() | Out-Null
    }

    # Save in-place; FileFormat=12 = wdFormatXMLDocument (.docx).
    $doc.SaveAs2($absPath, 12)

    Write-Host "[update-fields] TOCs=$tocCount fields=$fieldCount saved"
}
finally {
    if ($null -ne $doc) {
        try { $doc.Close($false) } catch { }
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null
    }
    if ($null -ne $word) {
        try { $word.Quit() } catch { }
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
