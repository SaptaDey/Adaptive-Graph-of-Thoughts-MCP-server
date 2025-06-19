import * as vscode from 'vscode';
import { parseNdjson } from './utils';

/**
 * Activates the extension by registering the 'agot.askGraph' command.
 *
 * When invoked, this command prompts the user for a question, sends it to a local server endpoint, and displays the parsed response in a new webview panel within Visual Studio Code.
 */
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

/**
 * Cleans up resources when the extension is deactivated.
 */
export function deactivate() {}
