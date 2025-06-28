import { McpError, ErrorCode } from '@modelcontextprotocol/sdk/types.js';
import { logger } from './logger.js';

export class ErrorHandler {
  static handleToolError(error, toolName, args) {
    const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    logger.error('Tool execution error', {
      errorId,
      toolName,
      error: error.message,
      stack: error.stack,
      args: JSON.stringify(args, null, 2),
    });

    if (error instanceof McpError) {
      return error;
    }

    // Handle specific error types
    if (error.code === 'ECONNREFUSED') {
      return new McpError(
        ErrorCode.InternalError,
        `Backend service unavailable. Please ensure the Adaptive Graph of Thoughts Python server is running. Error ID: ${errorId}`
      );
    }

    if (error.code === 'ETIMEDOUT') {
      return new McpError(
        ErrorCode.InternalError,
        `Request timeout. The operation took too long to complete. Error ID: ${errorId}`
      );
    }

    if (error.response) {
      const status = error.response.status;
      const message = error.response.data?.message || error.response.statusText;
      
      if (status === 400) {
        return new McpError(
          ErrorCode.InvalidParams,
          `Invalid request parameters: ${message}. Error ID: ${errorId}`
        );
      }
      
      if (status === 401 || status === 403) {
        return new McpError(
          ErrorCode.InvalidRequest,
          `Authentication or authorization failed: ${message}. Error ID: ${errorId}`
        );
      }
      
      if (status === 404) {
        return new McpError(
          ErrorCode.MethodNotFound,
          `Requested resource not found: ${message}. Error ID: ${errorId}`
        );
      }
      
      if (status >= 500) {
        return new McpError(
          ErrorCode.InternalError,
          `Backend service error (${status}): ${message}. Error ID: ${errorId}`
        );
      }
    }

    // Generic error fallback
    return new McpError(
      ErrorCode.InternalError,
      `Tool execution failed: ${error.message}. Error ID: ${errorId}`
    );
  }

  static handleConfigurationError(error) {
    logger.error('Configuration error', {
      error: error.message,
      stack: error.stack,
    });

    return new McpError(
      ErrorCode.InvalidRequest,
      `Configuration validation failed: ${error.message}`
    );
  }

  static handleValidationError(error, input) {
    logger.warn('Input validation error', {
      error: error.message,
      input: JSON.stringify(input, null, 2),
    });

    return new McpError(
      ErrorCode.InvalidParams,
      `Invalid input parameters: ${error.message}`
    );
  }

  static setupGlobalErrorHandlers(server) {
    // MCP server error handler
    server.onerror = (error) => {
      logger.error('MCP Server error', {
        error: error.message,
        stack: error.stack,
      });
    };

    // Process error handlers
    process.on('uncaughtException', (error) => {
      logger.error('Uncaught exception', {
        error: error.message,
        stack: error.stack,
      });
      
      // Give logger time to write before exiting
      setTimeout(() => {
        process.exit(1);
      }, 1000);
    });

    process.on('unhandledRejection', (reason, promise) => {
      logger.error('Unhandled promise rejection', {
        reason: reason instanceof Error ? reason.message : String(reason),
        stack: reason instanceof Error ? reason.stack : undefined,
        promise: promise.toString(),
      });
      
      // Give logger time to write before exiting
      setTimeout(() => {
        process.exit(1);
      }, 1000);
    });

    process.on('SIGINT', async () => {
      logger.info('Received SIGINT, shutting down gracefully');
      try {
        await server.close();
        logger.info('Server closed successfully');
      } catch (error) {
        logger.error('Error during server shutdown', {
          error: error.message,
          stack: error.stack,
        });
      }
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      logger.info('Received SIGTERM, shutting down gracefully');
      try {
        await server.close();
        logger.info('Server closed successfully');
      } catch (error) {
        logger.error('Error during server shutdown', {
          error: error.message,
          stack: error.stack,
        });
      }
      process.exit(0);
    });
  }
}

// Utility function to safely stringify objects for logging
export function safeStringify(obj, maxDepth = 3, currentDepth = 0) {
  if (currentDepth >= maxDepth) {
    return '[Max Depth Reached]';
  }

  if (obj === null) return 'null';
  if (obj === undefined) return 'undefined';
  if (typeof obj === 'string') return obj;
  if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);
  if (obj instanceof Error) {
    return {
      name: obj.name,
      message: obj.message,
      stack: obj.stack,
    };
  }

  if (Array.isArray(obj)) {
    return obj.map(item => safeStringify(item, maxDepth, currentDepth + 1));
  }

  if (typeof obj === 'object') {
    const result = {};
    for (const [key, value] of Object.entries(obj)) {
      // Skip potentially sensitive keys
      if (['password', 'token', 'key', 'secret'].some(sensitive => 
        key.toLowerCase().includes(sensitive)
      )) {
        result[key] = '[REDACTED]';
      } else {
        result[key] = safeStringify(value, maxDepth, currentDepth + 1);
      }
    }
    return result;
  }

  return String(obj);
}