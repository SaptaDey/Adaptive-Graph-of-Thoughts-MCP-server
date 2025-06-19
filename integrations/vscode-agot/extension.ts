import * as vscode from 'vscode';
import { parseNdjson } from './utils';

export function activate(context: vscode.ExtensionContext) {
    const disposable = vscode.commands.registerCommand('agot.askGraph', async () => {
        const question = await vscode.window.showInputBox({prompt: 'Ask Adaptive Graph of Thoughts'});
        if (!question) { return; }
        const res = await fetch('http://localhost:8000/nlq', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        const text = await res.text();
        const lines = parseNdjson(text);
        const pretty = JSON.stringify(lines, null, 2);
        const panel = vscode.window.createWebviewPanel('agotResult', 'AGoT Result', vscode.ViewColumn.One, {});
        panel.webview.html = `<pre>${pretty}</pre>`;
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}
