
$files = Get-ChildItem -Path "src" -Recurse -Filter "*.rs"

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content

    # Fix crate::domain::domain -> crate::domain
    $content = $content -replace "crate::domain::domain", "crate::domain"
    
    # Also handle possible triple domain if I messed up badly
    $content = $content -replace "crate::domain::domain::domain", "crate::domain"

    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Updated $($file.FullName)"
    }
}
