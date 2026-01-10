
$files = Get-ChildItem -Path "src" -Recurse -Filter "*.rs"

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content

    # Replace crate::math with crate::domain::math
    $content = $content -replace "crate::math", "crate::domain::math"
    
    # Replace crate::core with crate::domain::core
    # Note: excluding crate::core::domain if it exists (unlikely but safe to check)
    $content = $content -replace "crate::core", "crate::domain::core"

    # Fix double replacements if any (e.g. crate::domain::domain::math)
    $content = $content -replace "crate::domain::domain::math", "crate::domain::math"
    $content = $content -replace "crate::domain::domain::core", "crate::domain::core"

    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Updated $($file.FullName)"
    }
}
