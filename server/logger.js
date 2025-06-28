import { writeFileSync, appendFileSync, existsSync, mkdirSync, statSync, renameSync, readdirSync, unlinkSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class Logger {
  constructor(options = {}) {
    this.level = options.level || process.env.LOG_LEVEL || 'INFO';
    this.enableFile = options.enableFile !== false; // Enable file logging by default
    this.enableConsole = options.enableConsole !== false;
    this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB
    this.maxFiles = options.maxFiles || 5;
    
    this.logDir = options.logDir || join(__dirname, '..', 'logs');
    this.logFile = join(this.logDir, 'dxt-server.log');
    
    this.levels = {
      ERROR: 0,
      WARN: 1,
      INFO: 2,
      DEBUG: 3,
    };

    this.currentLevel = this.levels[this.level.toUpperCase()] || this.levels.INFO;
    
    this.initializeLogDirectory();
  }

  initializeLogDirectory() {
    if (this.enableFile && !existsSync(this.logDir)) {
      try {
        mkdirSync(this.logDir, { recursive: true });
      } catch (error) {
        console.error('Failed to create log directory:', error.message);
        this.enableFile = false;
      }
    }
  }

  formatMessage(level, message, meta = {}) {
    const timestamp = new Date().toISOString();
    const metaStr = Object.keys(meta).length > 0 ? ` ${JSON.stringify(meta)}` : '';
    return `[${timestamp}] ${level}: ${message}${metaStr}`;
  }

  writeToFile(formattedMessage) {
    if (!this.enableFile) return;

    try {
      // Simple rotation: if file is too large, archive it
      if (existsSync(this.logFile)) {
        const stats = statSync(this.logFile);
        if (stats.size > this.maxFileSize) {
          this.rotateLogFile();
        }
      }
      
      appendFileSync(this.logFile, formattedMessage + '\n');
    } catch (error) {
      console.error('Failed to write to log file:', error.message);
    }
  }

  rotateLogFile() {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const archiveFile = join(this.logDir, `dxt-server-${timestamp}.log`);
      renameSync(this.logFile, archiveFile);
      
      // Clean up old files
      this.cleanupOldLogs();
    } catch (error) {
      console.error('Failed to rotate log file:', error.message);
    }
  }

  cleanupOldLogs() {
    try {
      const files = readdirSync(this.logDir)
        .filter(file => file.startsWith('dxt-server-') && file.endsWith('.log'))
        .map(file => ({
          name: file,
          path: join(this.logDir, file),
          mtime: statSync(join(this.logDir, file)).mtime
        }))
        .sort((a, b) => b.mtime - a.mtime);

      // Remove excess files
      if (files.length > this.maxFiles) {
        files.slice(this.maxFiles).forEach(file => {
          try {
            unlinkSync(file.path);
          } catch (error) {
            console.error(`Failed to delete old log file ${file.name}:`, error.message);
          }
        });
      }
    } catch (error) {
      console.error('Failed to cleanup old logs:', error.message);
    }
  }

  log(level, message, meta = {}) {
    const levelValue = this.levels[level.toUpperCase()];
    if (levelValue === undefined || levelValue > this.currentLevel) {
      return;
    }

    const formattedMessage = this.formatMessage(level.toUpperCase(), message, meta);
    
    if (this.enableConsole) {
      // Write to stderr to avoid interfering with MCP stdio communication
      if (level.toUpperCase() === 'ERROR') {
        console.error(formattedMessage);
      } else {
        console.error(formattedMessage);
      }
    }
    
    this.writeToFile(formattedMessage);
  }

  error(message, meta = {}) {
    this.log('ERROR', message, meta);
  }

  warn(message, meta = {}) {
    this.log('WARN', message, meta);
  }

  info(message, meta = {}) {
    this.log('INFO', message, meta);
  }

  debug(message, meta = {}) {
    this.log('DEBUG', message, meta);
  }

  // Create a child logger with additional context
  child(context = {}) {
    const childLogger = Object.create(this);
    childLogger.defaultMeta = { ...this.defaultMeta, ...context };
    childLogger.log = (level, message, meta = {}) => {
      this.log(level, message, { ...childLogger.defaultMeta, ...meta });
    };
    return childLogger;
  }
}

// Export singleton instance
export const logger = new Logger();
export { Logger };