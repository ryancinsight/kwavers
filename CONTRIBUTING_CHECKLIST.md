# Contributing to kwavers - Checklist

This document provides a checklist for contributors to ensure consistency, quality, and maintainability of the `kwavers` codebase. Whether you're adding a new feature, fixing a bug, or improving documentation, please review this list.

## 1. General Guidelines

*   [ ] **Fork the Repository:** Create your own fork of the `kwavers` repository to work on your changes.
*   [ ] **Create a New Branch:** Create a new branch from `main` (or the relevant development branch) for your work. Use a descriptive branch name (e.g., `feature/elastic-pml`, `bugfix/pml-stability`).
*   [ ] **Adhere to Code Style:**
    *   Run `cargo fmt` to format your Rust code according to the project's style.
    *   Follow standard Rust naming conventions (e.g., `snake_case` for functions and variables, `CamelCase` for types).
*   [ ] **Keep Commits Atomic and Descriptive:** Make small, logical commits with clear and concise commit messages.
*   [ ] **Update `AGENTS.md` (if applicable):** If your changes affect how an AI agent should interact with or understand the codebase, update any relevant `AGENTS.md` files.
*   [ ] **Consider Performance:** For performance-critical sections, profile your changes and ensure they do not introduce significant regressions. Add comments explaining performance-related design choices.
*   [ ] **Stay Up-to-Date:** Regularly rebase your branch on the latest `main` (or development branch) to avoid large merge conflicts.

## 2. For New Features

*   [ ] **Issue Discussion (Recommended):** Before starting significant work, consider opening an issue to discuss the proposed feature, its design, and implementation plan.
*   [ ] **Design:**
    *   Ensure the new feature integrates well with the existing architecture.
    *   Consider modularity and extensibility.
    *   Identify any new dependencies and justify their inclusion.
*   [ ] **Implementation:**
    *   Write clear, maintainable, and well-commented code.
    *   Add comments for complex logic or non-obvious design choices.
*   [ ] **Unit Tests:**
    *   Add comprehensive unit tests for all new public functions and methods.
    *   Ensure tests cover a range of valid inputs, edge cases, and error conditions.
    *   Aim for high test coverage for the new code.
*   [ ] **Integration Tests / Examples:**
    *   If applicable, add an integration test or a new example in the `examples/` directory to demonstrate the feature and ensure it works correctly with other parts of the system.
    *   The example should be runnable and produce meaningful output (e.g., CSV data, plots if a plotting utility is part of the example).
*   [ ] **Documentation:**
    *   Add Rustdoc comments for all new public structs, enums, traits, functions, and methods.
    *   Explain what the code does, its parameters, return values, and any panics or errors.
    *   Provide usage examples in the documentation where appropriate.
*   [ ] **Update `README.md` / PRD:** If the feature is significant, update the `README.md` (e.g., Key Features) and potentially the `PRODUCT_REQUIREMENTS.md` to reflect its inclusion.
*   [ ] **All Tests Pass:** Ensure `cargo test --all-features` (if features are used) passes locally before submitting a Pull Request.
*   [ ] **No New Warnings:** Ensure your changes do not introduce new compiler warnings (`cargo clippy --all-targets --all-features -- -D warnings`).

## 3. For Bug Fixes

*   [ ] **Link to Issue (Recommended):** If the bug fix addresses an existing issue, reference the issue number in your commit messages and Pull Request.
*   [ ] **Minimal and Focused Change:** Ensure the fix is targeted to the specific bug and avoids unrelated changes.
*   [ ] **Regression Test:**
    *   Add a new test case (unit or integration) that specifically reproduces the bug.
    *   Verify that this test fails *before* your fix and passes *after* your fix. This ensures the bug is truly fixed and prevents regressions.
*   [ ] **Documentation (if applicable):** If the bug was due to misleading documentation or API misuse, update the relevant documentation.
*   [ ] **All Tests Pass:** Ensure `cargo test --all-features` passes locally.
*   [ ] **No New Warnings:** Ensure your changes do not introduce new compiler warnings.

## 4. For Documentation Changes

*   [ ] **Clarity and Accuracy:** Ensure documentation is clear, concise, accurate, and up-to-date.
*   [ ] **Grammar and Spelling:** Proofread your changes.
*   [ ] **Build Documentation Locally:** Run `cargo doc --open` to preview your documentation changes.
*   [ ] **Update Examples (if necessary):** If API documentation changes, ensure any relevant examples are also updated.

## 5. Submitting a Pull Request (PR)

*   [ ] **Descriptive PR Title and Description:**
    *   Clearly explain the purpose of the PR and the changes made.
    *   Reference any related issues (e.g., "Fixes #123").
*   [ ] **Self-Review:** Review your own changes one last time before submitting.
*   [ ] **Request Reviewers:** Request reviews from appropriate maintainers or team members.
*   [ ] **Address Feedback:** Be responsive to review comments and address any requested changes.

By following this checklist, you help us maintain a high-quality, robust, and easy-to-understand codebase for `kwavers`. Thank you for your contributions!
