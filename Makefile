MAKEFLAGS += --no-print-directory

.PHONY: _login-to-github
.PHONY: _trigger-act-job-with-github-token

# prefixed with _ to prevent these commands from showing up in make's autocomplete
_login-to-github:
	gh auth login -s $(SCOPES)
_trigger-act-job-with-github-token:
	@act -j ${JOB} -s GITHUB_TOKEN="$$(gh auth token)"

act-login-to-github-and-%:
	@make _login-to-github SCOPES=${SCOPES}
	@make act-$*

act-login-to-github-and-build-devcontainer: SCOPES=read:packages
act-login-to-github-and-build-and-publish-devcontainer: SCOPES=write:packages

act-%:
	@make _trigger-act-job-with-github-token JOB="$*"

# ensures these commands are displayed in make's autocomplete
act-build-devcontainer :
act-build-and-publish-devcontainer :
