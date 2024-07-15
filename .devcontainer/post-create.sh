#!/bin/sh

VSCODE_HOME=/root

if [ -d /user-home/.ssh ]; then
    echo "Making user ssh available in container..."
    mkdir -p $VSCODE_HOME/.ssh
    chmod 0700 $VSCODE_HOME/.ssh
    for f in /user-home/.ssh/*
    do
        cp "$f" $VSCODE_HOME/.ssh/"$(basename "$f")"
        chmod 0600 $VSCODE_HOME/.ssh/"$(basename "$f")"
    done
fi

# If the user has a git config file, copy it
if [ -f /user-home/.gitconfig ]; then
    echo "Copying user .gitconfig..."
    cp /user-home/.gitconfig $VSCODE_HOME/.gitconfig
    echo "Enabling HTTP use path, in case the user cloned with HTTP"
    git config --global credential.useHttpPath true
fi
