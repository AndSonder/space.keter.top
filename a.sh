#!/bin/bash
# git filter-branch -f --env-filter '
# if [ "$GIT_AUTHOR_NAME" = "coronapolvo" ]
# then
# export GIT_AUTHOR_NAME="andsonder"
# export GIT_AUTHOR_EMAIL="changlu@keter.top"
# fi
# ' HEAD

git filter-branch -f --env-filter '
if [ "$GIT_COMMITTER_NAME" = "coronapolvo" ]
then
export GIT_COMMITTER_NAME="andsonder"
export GIT_COMMITTER_EMAIL="changlu@keter.top"
fi
' HEAD