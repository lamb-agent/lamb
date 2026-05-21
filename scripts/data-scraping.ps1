function Get-BenchmarkMetrics {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true, Position = 0)]
        [System.IO.DirectoryInfo[]]$InputDirs,

        [Parameter(Mandatory = $false)]
        [string]$OutputFileName = "metrics_report.csv",

        [Parameter(Mandatory = $false)]
        [string]$OutputDir = (Get-Location).Path
    )

    process {
        # Define the 4 target suites
        $Suites = @("banking", "travel", "workspace", "slack", "coding")

        # Define patterns to identify the 3 categories based on filenames
        $Categories = @{
            "User Tasks"      = "^user-user_task_\d+\.json$"
            "Injection Tasks" = "^injection-injection_task_\d+\.json$"
            "Attacks"         = "^attack-user_task_\d+-injection_task_\d+\.json$"
        }

        # Initialize array to store results
        $Report = @()

        # Helper function to count regex matches across an array of file contents
        function Get-MatchCount {
            param (
                [string[]]$JsonDumps,  # Explicitly accepts an array of strings
                [string]$Pattern
            )
            if (-not $JsonDumps) { return 0 }

            $TotalMatches = 0
            # Loop through each individual file's content
            foreach ($Dump in $JsonDumps) {
                if ([string]::IsNullOrEmpty($Dump)) { continue }
                $TotalMatches += ([regex]::Matches($Dump, $Pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
            }
            return $TotalMatches
        }

        # Loop through each directory provided via -InputDirs
        foreach ($BenchmarkDir in $InputDirs) {

            # Loop through each suite inside the benchmark folder
            foreach ($Suite in $Suites) {
                $SuitePath = Join-Path $BenchmarkDir.FullName $Suite

                if (-not (Test-Path $SuitePath)) { continue } # Skip if suite folder missing

                # Get all relevant individual JSON files (excluding results.json)
                $JsonFiles = Get-ChildItem -Path $SuitePath -Filter "*.json" | Where-Object { $_.Name -ne "results.json" }

                # Process each category
                foreach ($CatName in $Categories.Keys) {
                    $Pattern = $Categories[$CatName]

                    # Filter files belonging to current category
                    $CatFiles = $JsonFiles | Where-Object { $_.Name -match $Pattern }

                    $Count = $CatFiles.Count
                    $UtilityTrueCount = 0
                    $SecurityTrueCount = 0
                    $TimeoutCount = 0
                    $ConnectionCount = 0
                    $ErrOtherCount = 0

                    if ($Count -gt 0) {
                        # This creates an array of strings, one string per file
                        $JsonDumps = $CatFiles | ForEach-Object { Get-Content -Path $_.FullName -Raw }

                        # Scan files for the target strings efficiently
                        $UtilityPattern = '"utility":\s*true(?=[^[\]{}]*\s*\}\s*\z)'
                        $UtilityTrueCount = Get-MatchCount -JsonDumps $JsonDumps -Pattern $UtilityPattern

                        # Only check security for Attacks category
                        if ($CatName -eq "Attacks") {
                            $SecurityPattern = '"security":\s*true(?=[^[\]{}]*\s*\}\s*\z)'
                            $SecurityTrueCount = Get-MatchCount -JsonDumps $JsonDumps -Pattern $SecurityPattern
                        }

                        $TimeoutCount = Get-MatchCount -JsonDumps $JsonDumps -Pattern '"error":\s*"Request timed out."\s*}\s*\z'
                        $ConnectionCount = Get-MatchCount -JsonDumps $JsonDumps -Pattern '"error":\s*"Connection error."\s*}\s*\z'

                        $AllErrorsPattern = '"error":\s*"[^"\s]+[^"]*"\s*}\s*\z'
                        $TotalErrors = Get-MatchCount -JsonDumps $JsonDumps -Pattern $AllErrorsPattern
                        $ErrOtherCount = [Math]::Max(0, $TotalErrors - $TimeoutCount - $ConnectionCount)
                    }

                    # Append detailed row
                    $Report += [PSCustomObject]@{
                        "Benchmark"        = $BenchmarkDir.Name
                        "Suite"            = $Suite
                        "Category"         = $CatName
                        "Count"            = $Count
                        "Utility_True"     = $UtilityTrueCount
                        "Security_True"    = $SecurityTrueCount
                        "Error_Timeout"    = $TimeoutCount
                        "Error_Connection" = $ConnectionCount
                        "Error_Other"      = $ErrOtherCount
                    }
                }
            }
        }

        # Resolve final export path
        $FinalOutputPath = Join-Path $OutputDir $OutputFileName

        # Export data directly to a CSV file ready for Excel
        $Report | Export-Csv -Path $FinalOutputPath -NoTypeInformation -Encoding utf8 -Append

        Write-Host "Report successfully exported to: $FinalOutputPath" -ForegroundColor Green
    }
}