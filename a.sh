#!/bin/bash
git filter-branch -f --env-filter "
GIT_AUTHOR_NAME='andsonder';
GIT_AUTHOR_EMAIL='changlu@keter.top';
GIT_COMMITTER_NAME='andsonder';
GIT_COMMITTER_EMAIL='changlu@keter.top'
" HEAD