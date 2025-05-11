<#
.SYNOPSIS
    Automates staging, committing, and pushing changes to a Git repository.
.DESCRIPTION
    This script stages all current changes, prompts the user for a commit message,
    creates a commit, and then pushes the commit to the 'origin' remote
    on the 'main' branch. It includes enhanced error reporting and pauses at the end.
.NOTES
    Author: Gemini
    Date: 2025-05-11
    Ensure you are in the root directory of your Git repository before running.
    Modify the branch name in `git push origin main` if your primary branch is different (e.g., master).
#>

# Script Start
Write-Host "Starting Git update script..." -ForegroundColor Cyan

# Check Git Status first
Write-Host "Checking git status..." -ForegroundColor Yellow
git status
Write-Host "------------------------------------"

try {
    Write-Host "Staging all changes..." -ForegroundColor Yellow
    git add .
    # Check if there are changes to commit
    $statusOutput = git status --porcelain
    if (-not $statusOutput) {
        Write-Host "No changes to commit." -ForegroundColor Green
        # We'll let it try to push existing commits if any.
    } else {
        Write-Host "All changes staged." -ForegroundColor Green

        # Prompt for the commit message
        $commitMessage = Read-Host "Enter your commit message"

        # Check if the user actually entered a message
        if ([string]::IsNullOrWhiteSpace($commitMessage)) {
            Write-Error "Commit message cannot be empty. Aborting commit."
            # Optionally, you could loop here to re-prompt, or just exit.
            # For simplicity, we'll throw an error that the catch block can handle.
            throw "Commit message was empty."
        }

        Write-Host "Committing changes with message: '$commitMessage'" -ForegroundColor Yellow
        git commit -m $commitMessage
        Write-Host "Changes committed." -ForegroundColor Green
    }

    Write-Host "Pushing changes to remote repository (origin main)..." -ForegroundColor Yellow
    # Adjust 'main' to your primary branch name if different (e.g., 'master')
    git push origin main
    # Check the exit code of the last command (git push)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Git push failed. See messages above. You might need to 'git pull' first."
        # The script will continue to the finally block, but this gives a more direct error.
    } else {
        Write-Host "Changes pushed successfully." -ForegroundColor Green
    }

    Write-Host "Git update process completed." -ForegroundColor Cyan
}
catch {
    Write-Error "An error occurred during the Git process: $($_.Exception.Message)"
    Write-Host "Script execution failed. See error message above." -ForegroundColor Red
    # You can add more detailed error information if needed:
    # Write-Host "Error Details: $($_.ToString())" -ForegroundColor Red
    # Write-Host "Stack Trace: $($_.ScriptStackTrace)" -ForegroundColor Red
}
finally {
    # This block will execute whether the try block succeeded or failed.
    Write-Host "------------------------------------"
    Write-Host "Script execution finished."
    Read-Host "Press Enter to exit..."
}

# End of script
