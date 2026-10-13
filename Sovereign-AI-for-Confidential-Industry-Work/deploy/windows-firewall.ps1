<#
.SYNOPSIS
    Arms, disarms or reports the per-application egress block for Kavach.

.DESCRIPTION
    The Windows counterpart of deploy/nftables.conf, and it exists because of a
    question with an obvious answer once asked out loud: the claim is that THIS
    APPLICATION cannot reach the network, not that the computer has no network.
    Nobody unplugs a refinery workstation to run a workbench on it.

    So this adds exactly one outbound BLOCK rule, scoped to one executable - the
    Python interpreter that runs the gateway. Everything else on the machine
    keeps working: the browser that opens the UI, Docker pulling nothing, the
    operator's mail. Fire the canary afterwards and the refusal it reports is
    then a POLICY refusing this program, which is the only thing worth showing.

    Without it, the canary probes an RFC 5737 documentation address and reports
    whatever the routing table happens to say. Measured on a development box with
    no rule armed: the documentation address times out (proving nothing) while
    1.1.1.1 answers in 5 ms - the application had full internet the whole time.

.NOTES
    Requires an elevated PowerShell. Blocking traffic is an administrative act
    and there is no way around that; the script refuses rather than half-working.

    The rule is REMOVABLE and this script is the way to remove it (-Disarm).
    Nothing else on the box is touched - no profile is turned off, no other rule
    is edited, and the machine's own connectivity is never changed.

.EXAMPLE
    .\deploy\windows-firewall.ps1 -Status
    .\deploy\windows-firewall.ps1 -Arm
    .\deploy\windows-firewall.ps1 -Disarm
#>

[CmdletBinding(DefaultParameterSetName = 'Status')]
param(
    [Parameter(ParameterSetName = 'Arm')]    [switch]$Arm,
    [Parameter(ParameterSetName = 'Disarm')] [switch]$Disarm,
    [Parameter(ParameterSetName = 'Status')] [switch]$Status,

    # The executable to contain. Defaults to this repository's venv interpreter,
    # which is what `start.bat` launches and therefore what opens the sockets.
    [string]$Program
)

$ErrorActionPreference = 'Stop'

# Must match services/kavach/sovereignty/enforcement.py:WINDOWS_RULE_NAME. The
# detector matches this name exactly, so a rename here without a rename there
# leaves the panel reporting "nothing is armed" over a machine that is.
$RuleName = 'Kavach egress block'

function Resolve-Program {
    param([string]$Given)
    if ($Given) { return (Resolve-Path -LiteralPath $Given).Path }
    $repo = Split-Path -Parent $PSScriptRoot
    $venv = Join-Path $repo '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) { return (Resolve-Path -LiteralPath $venv).Path }
    throw "no .venv found at $venv - pass -Program with the python.exe that runs the gateway"
}

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Show-Status {
    # netsh rather than Get-NetFirewallRule: same reason the detector uses it -
    # it is present on every install and needs no module or execution policy.
    $out = & netsh advfirewall firewall show rule "name=$RuleName" dir=out verbose 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or $out -match 'No rules match') {
        Write-Host "NOT ARMED - this application can reach the network like any other program." -ForegroundColor Yellow
        Write-Host "            The canary can only report what the routing table says, which is not proof."
        Write-Host "            Arm it with:  .\deploy\windows-firewall.ps1 -Arm   (as Administrator)"
        return $false
    }
    Write-Host "ARMED" -ForegroundColor Green
    Write-Host $out.Trim()
    return $true
}

switch ($PSCmdlet.ParameterSetName) {

    'Arm' {
        if (-not (Test-Admin)) {
            throw "this needs an elevated PowerShell - adding a firewall rule is an administrative act. Right-click PowerShell -> Run as administrator, then run this again."
        }
        $exe = Resolve-Program -Given $Program

        # Remove first so re-arming after moving the venv cannot leave a stale
        # rule pointing at an executable that no longer exists - which would
        # report ARMED while containing nothing.
        & netsh advfirewall firewall delete rule "name=$RuleName" dir=out | Out-Null

        & netsh advfirewall firewall add rule `
            "name=$RuleName" `
            dir=out `
            action=block `
            "program=$exe" `
            enable=yes `
            profile=any `
            "description=Kavach sovereignty: this workbench may not reach the network. Removable with deploy\windows-firewall.ps1 -Disarm." | Out-Null

        if ($LASTEXITCODE -ne 0) { throw "netsh refused to add the rule (exit $LASTEXITCODE)" }

        Write-Host "ARMED - outbound traffic from this program is now blocked:" -ForegroundColor Green
        Write-Host "  $exe"
        Write-Host ""
        Write-Host "The rest of this machine is untouched and still online."
        Write-Host "Restart the gateway, then press 'Fire the canary' on the Offline proof page:"
        Write-Host "  a refusal there is now this policy refusing this program."
    }

    'Disarm' {
        if (-not (Test-Admin)) { throw "this needs an elevated PowerShell." }
        & netsh advfirewall firewall delete rule "name=$RuleName" dir=out | Out-Null
        Write-Host "DISARMED - the rule has been removed. Nothing else was changed." -ForegroundColor Yellow
    }

    default { [void](Show-Status) }
}
