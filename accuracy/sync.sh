#!/bin/bash
# copy accuracy notes into the experiments worktree, commit, push in background
P=/workspace/projects/fc66e2e1-bc5c-4c41-b381-c3c20e2d0eca
cd $P/moonboost-exp && mkdir -p accuracy/logs && cp $P/notes/acc/*.md $P/notes/acc/*.py $P/notes/acc/*.sh accuracy/ 2>/dev/null
[ -d $P/notes/acc/logs ] && cp $P/notes/acc/logs/* accuracy/logs/ 2>/dev/null
git add -A && git -c user.name=Juwan -c user.email=180104975+Juwan-Hwang@users.noreply.github.com commit -qm "${1:-Accuracy experiments}" && (git push -q origin experiments > /tmp/exp-push.log 2>&1 &)
git log --oneline -1
