<#
.SYNOPSIS
    Automates staging, committing, and pushing changes to a Git repository.
.DESCRIPTION
    This script stages all current changes, creates a commit with a predefined
    detailed message summarizing recent development work, and then pushes
    the commit to the 'origin' remote on the 'main' branch.
    It includes enhanced error reporting and pauses at the end.
.NOTES
    Author: Gemini
    Date: 2025-05-11
    Ensure you are in the root directory of your Git repository before running.
    Modify the branch name in `git push origin main` if your primary branch is different (e.g., master).
#>

# Define the detailed commit message
$commitMessage = @"
Refactor: Core Agent Logic, Command Parsing, and Feature Enhancements (May 11)

This commit includes several key improvements and bug fixes developed today:

1.  **Agent `run` Method Signature (`TypeError` Fix)**:
    * Modified `app.py` and `main.py` to conditionally pass the
      `associated_goal_id_for_new_plan` argument only to the
      `PlannerAgent`. This resolved a `TypeError` that occurred when
      this argument was incorrectly passed to other agents not expecting it.

2.  **`app.py` Robustness (`NameError` Fix for `VectorMemory`)**:
    * Resolved a `NameError` in `app.py` that occurred during Flask
      auto-reloads. The error was related to the type hint for `VectorMemory`
      not being defined if its import failed on a reload. Fixed by using
      a string literal for the type hint and adjusting the import and
      instantiation logic for `VectorMemory`.

3.  **AnalystAgent Tool Usage Enhancement**:
    * Improved the `AnalystAgent`'s ability to parse and execute tool
      commands (formatted as JSON). It can now correctly extract and
      process tool requests even if they are embedded within surrounding
      explanatory text from the LLM.

4.  **ResearcherAgent ReAct Loop & Stage Advancement**:
    * Significantly refactored the `ResearcherAgent.run()` method to
      implement a more robust iterative ReAct (Reason-Act-Observe) loop.
      This allows the agent to perform multi-step research by iteratively
      calling the LLM, executing tools, and using observations.
    * Corrected and enhanced stage advancement logic to ensure the agent
      progresses through planning, information gathering, analysis, and
      synthesis stages more reliably.

5.  **PlannerAgent Command Recognition**:
    * Addressed issues in `PlannerAgent.run()` where commands like
      "create and execute plan for..." were not being recognized.
    * Implemented thorough whitespace normalization (including handling
      non-breaking spaces like `\xa0`) at the beginning of command processing.
    * Refined regex patterns and string matching logic for more flexible
      and accurate command parsing, ensuring goal text is correctly
      extracted while preserving its original casing for the LLM.

6.  **Goal Deletion Feature (Backend Implementation)**:
    * Added a `delete_goal_permanently` method to `EnhancedMemory`
      (and by extension `VectorMemory`) to allow for the complete removal
      of goals from both active and completed lists, including their
      associated memory segments.
    * Implemented a corresponding command handler in `QuartermasterAgent`
      (`delete goal permanently by id [goal_id]`) to expose this functionality.
    * Note: Web UI integration for this permanent deletion feature is still pending.

These changes collectively aim to improve the stability of agent execution,
enhance the reliability of complex, multi-step tasks (especially for research
and planning), and introduce foundational work for better goal management.
"@

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
        # You might want to exit here or let it proceed to push (if there are unpushed commits)
        # For now, we'll let it try to push existing commits.
    } else {
        Write-Host "All changes staged." -ForegroundColor Green

        Write-Host "Committing changes..." -ForegroundColor Yellow
        # Use --quiet to suppress commit output if $commitMessage is very long and makes it hard to see other messages
        # git commit -m $commitMessage --quiet
        git commit -m $commitMessage
        Write-Host "Changes committed." -ForegroundColor Green
    }

    Write-Host "Pushing changes to remote repository (origin main)..." -ForegroundColor Yellow
    # Adjust 'main' to your primary branch name if different (e.g., 'master')
    git push origin main
    Write-Host "Changes pushed successfully." -ForegroundColor Green

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
