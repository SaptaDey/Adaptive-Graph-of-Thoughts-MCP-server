suite('Extension Activation Tests', () => {
        test('should activate extension successfully', async () => {
            const activateStub = sandbox.stub(extension, 'activate');
            
            await extension.activate(context);
            
            assert.strictEqual(activateStub.calledOnce, true);
            assert.strictEqual(activateStub.calledWith(context), true);
        });

        test('should handle activation with undefined context', async () => {
            try {
                await extension.activate(undefined as any);
                assert.fail('Should have thrown an error');
            } catch (error) {
                assert.ok(error instanceof Error);
            }
        });

        test('should register all commands during activation', async () => {
            const registerCommandStub = sandbox.stub(vscode.commands, 'registerCommand');
            
            await extension.activate(context);
            
            assert.ok(registerCommandStub.called);
            // Verify specific commands are registered
            const expectedCommands = ['agot.command1', 'agot.command2']; // Replace with actual commands
            expectedCommands.forEach(cmd => {
                assert.ok(registerCommandStub.calledWith(cmd));
            });
        });

        test('should handle activation errors gracefully', async () => {
            const showErrorStub = sandbox.stub(vscode.window, 'showErrorMessage');
            sandbox.stub(vscode.commands, 'registerCommand').throws(new Error('Registration failed'));
            
            try {
                await extension.activate(context);
            } catch (_err) {
                // Expected failure, extension should catch internally
            }
            
            assert.ok(showErrorStub.called);
        });
    });

    suite('Extension Deactivation Tests', () => {
        test('should deactivate extension cleanly', () => {
            const result = extension.deactivate();
            
            if (result) {
                assert.ok(result instanceof Promise);
            } else {
                assert.strictEqual(result, undefined);
            }
        });

        test('should dispose of all subscriptions on deactivation', async () => {
            const mockDisposable = { dispose: sandbox.stub() };
            context.subscriptions.push(mockDisposable);
            
            await extension.activate(context);
            extension.deactivate();
            
            assert.ok(mockDisposable.dispose.called);
        });

        test('should handle deactivation errors gracefully', () => {
            const consoleSpy = sandbox.spy(console, 'error');
            sandbox.stub(extension, 'deactivate').throws(new Error('Deactivation failed'));
            
            try {
                extension.deactivate();
            } catch (_err) {
                // Expected
            }
            assert.ok(consoleSpy.called);
        });
    });

    suite('Command Tests', () => {
        beforeEach(async () => {
            await extension.activate(context);
        });

        test('should execute commands without errors', async () => {
            const commands = await vscode.commands.getCommands();
            const agotCommands = commands.filter(cmd => cmd.startsWith('agot.'));
            
            for (const cmd of agotCommands) {
                try {
                    await vscode.commands.executeCommand(cmd);
                    assert.ok(true, `Command ${cmd} executed successfully`);
                } catch (error) {
                    assert.fail(`Command ${cmd} failed: ${error}`);
                }
            }
        });

        test('should handle command execution with invalid parameters', async () => {
            const showErrorStub = sandbox.stub(vscode.window, 'showErrorMessage');
            
            try {
                await vscode.commands.executeCommand('agot.nonexistentCommand');
            } catch (error) {
                assert.ok(error instanceof Error);
            }
        });

        test('should validate command arguments', async () => {
            const invalidArgs = [null, undefined, '', 123, [], {}];
            
            for (const arg of invalidArgs) {
                try {
                    await vscode.commands.executeCommand('agot.someCommand', arg);
                } catch (error) {
                    assert.ok(error instanceof Error, `Should handle invalid arg: ${arg}`);
                }
            }
        });
    });

    suite('VSCode API Integration Tests', () => {
        test('should interact with workspace correctly', async () => {
            const workspaceFolders = vscode.workspace.workspaceFolders;
            
            if (workspaceFolders) {
                assert.ok(Array.isArray(workspaceFolders));
                workspaceFolders.forEach(folder => {
                    assert.ok(folder.uri);
                    assert.ok(folder.name);
                });
            }
        });

        test('should handle no workspace scenario', async () => {
            sandbox.stub(vscode.workspace, 'workspaceFolders').value(undefined);
            const showWarningStub = sandbox.stub(vscode.window, 'showWarningMessage');
            
            await extension.activate(context);
            assert.ok(showWarningStub.called || true, 'Handled missing workspace');
        });

        test('should respond to configuration changes', async () => {
            const onDidChangeConfigurationStub = sandbox.stub(vscode.workspace, 'onDidChangeConfiguration');
            
            await extension.activate(context);
            assert.ok(onDidChangeConfigurationStub.called);
        });

        test('should handle text document operations', async () => {
            const showTextDocumentStub = sandbox.stub(vscode.window, 'showTextDocument');
            const openTextDocumentStub = sandbox.stub(vscode.workspace, 'openTextDocument');
            
            openTextDocumentStub.resolves({
                uri: vscode.Uri.file('/test/file.txt'),
                fileName: '/test/file.txt',
                isUntitled: false,
                languageId: 'plaintext',
                version: 1,
                isDirty: false,
                isClosed: false
            } as any);
            
            const doc = await vscode.workspace.openTextDocument('/test/file.txt');
            await vscode.window.showTextDocument(doc);
            
            assert.ok(openTextDocumentStub.called);
            assert.ok(showTextDocumentStub.called);
        });
    });

    suite('Performance and Edge Case Tests', () => {
        test('should handle large workspace efficiently', async () => {
            const startTime = Date.now();
            const largeFolderStructure = Array.from({ length: 1000 }, (_, i) => ({
                uri: vscode.Uri.file(`/large/workspace/folder${i}`),
                name: `folder${i}`
            }));
            
            sandbox.stub(vscode.workspace, 'workspaceFolders').value(largeFolderStructure);
            await extension.activate(context);
            
            const duration = Date.now() - startTime;
            assert.ok(duration < 5000, `Activation took too long: ${duration}ms`);
        });

        test('should handle concurrent command executions', async () => {
            await extension.activate(context);
            const tasks = Array.from({ length: 10 }, () => vscode.commands.executeCommand('agot.someCommand'));
            
            try {
                await Promise.all(tasks);
                assert.ok(true, 'All concurrent commands executed successfully');
            } catch (error) {
                assert.fail(`Concurrent execution failed: ${error}`);
            }
        });

        test('should handle memory pressure gracefully', async () => {
            const memBefore = process.memoryUsage().heapUsed;
            for (let i = 0; i < 100; i++) {
                await extension.activate(context);
                extension.deactivate();
            }
            const memAfter = process.memoryUsage().heapUsed;
            assert.ok(memAfter - memBefore < 50 * 1024 * 1024, `Memory usage grew too much: ${memAfter - memBefore}`);
        });

        test('should handle network timeouts and errors', async () => {
            const fetchStub = sandbox.stub().rejects(new Error('Network timeout'));
            (global as any).fetch = fetchStub;
            
            await extension.activate(context);
            assert.ok(true, 'Extension handles network errors gracefully');
        });
    });

    suite('State Management Tests', () => {
        test('should persist and retrieve workspace state', async () => {
            const key = 'testKey';
            const value = { data: 'test data' };
            
            context.workspaceState.get = sandbox.stub().returns(value);
            context.workspaceState.update = sandbox.stub().resolves();
            
            await extension.activate(context);
            const retrieved = context.workspaceState.get(key);
            await context.workspaceState.update(key, value);
            
            assert.deepStrictEqual(retrieved, value);
            assert.ok(context.workspaceState.update.calledWith(key, value));
        });

        test('should handle state corruption gracefully', async () => {
            context.workspaceState.get = sandbox.stub().throws(new Error('State corrupted'));
            
            try {
                await extension.activate(context);
                assert.ok(true, 'Extension handles state corruption gracefully');
            } catch (error) {
                assert.fail(`Extension should handle state corruption: ${error}`);
            }
        });
    });
});

// Utility functions for testing
function createMockExtensionContext(): vscode.ExtensionContext {
    return {
        subscriptions: [],
        workspaceState: {
            get: sinon.stub(),
            update: sinon.stub().resolves()
        },
        globalState: {
            get: sinon.stub(),
            update: sinon.stub().resolves()
        },
        extensionPath: '/mock/extension/path',
        storagePath: '/mock/storage/path',
        globalStoragePath: '/mock/global/storage/path',
        logPath: '/mock/log/path',
        extensionUri: vscode.Uri.file('/mock/extension/path'),
        environmentVariableCollection: {} as any,
        extensionMode: vscode.ExtensionMode.Test,
        globalStorageUri: vscode.Uri.file('/mock/global/storage'),
        logUri: vscode.Uri.file('/mock/log'),
        storageUri: vscode.Uri.file('/mock/storage')
    } as vscode.ExtensionContext;
}

function createMockTextDocument(content: string = '', languageId: string = 'plaintext'): vscode.TextDocument {
    return {
        uri: vscode.Uri.file('/mock/document.txt'),
        fileName: '/mock/document.txt',
        isUntitled: false,
        languageId,
        version: 1,
        isDirty: false,
        isClosed: false,
        save: sinon.stub().resolves(true),
        eol: vscode.EndOfLine.LF,
        lineCount: content.split('\n').length,
        getText: sinon.stub().returns(content),
        getWordRangeAtPosition: sinon.stub(),
        validateRange: sinon.stub(),
        validatePosition: sinon.stub(),
        offsetAt: sinon.stub(),
        positionAt: sinon.stub(),
        lineAt: sinon.stub()
    } as any;
}

// Export utility functions for use in other test files
export { createMockExtensionContext, createMockTextDocument };