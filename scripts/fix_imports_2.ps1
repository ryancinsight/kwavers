
$files = Get-ChildItem -Path "src" -Recurse -Filter "*.rs"

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content

    # Fix core::error -> domain::core::error
    $content = $content -replace "core::error", "domain::core::error"
    
    # Fix core::constants -> domain::core::constants
    $content = $content -replace "core::constants", "domain::core::constants"

    # Fix core::time -> domain::core::time (just in case)
    $content = $content -replace "core::time", "domain::core::time"

    # Handle `core::{` followed by `constants` or `error` commonly seen
    # This is harder with regex, but let's fix the explicit ones first.

    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Updated $($file.FullName)"
    }
}
