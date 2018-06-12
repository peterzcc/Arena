#!/usr/bin/env zsh

for _pane in $(tmux list-windows -F '#I'); do
    tmux send-keys -t ${_pane} "Enter"
done