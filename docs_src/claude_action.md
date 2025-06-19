# Claude Desktop Custom Action

A lightweight custom action lets Claude Desktop query the Adaptive Graph of Thoughts server without any code.

Place the `manifest.json` and `index.js` files from `integrations/claude-action` into Claude's custom actions directory. When text is selected, choose **Ask Adaptive Graph** from the action menu and the selection will be sent to the `/nlq` endpoint. The returned summary is displayed inside Claude.
