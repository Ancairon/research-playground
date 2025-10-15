/**
 * Netdata Chart Predictions Injector
 * 
 * Drop-in script to inject hardcoded prediction values into chart data.
 * Usage:
 *   1. Open your Netdata dashboard in browser (http://localhost:19999)
 *   2. Open browser DevTools console (F12)
 *   3. Paste this entire script and press Enter
 *   4. Navigate to a chart (or refresh) - predictions will appear on the right side
 * 
 * Target: Inject predictions for ONE chart (configurable below)
 */

(function() {
  'use strict';
  
  // ========== CONFIGURATION ==========
  const CONFIG = {
    // Chart to augment with predictions (use chart ID, e.g., 'system.cpu', 'system.ram', etc.)
    targetChartId: 'system.cpu',
    
    // Filter to specific dimension(s) - if empty, all dimensions are augmented
    targetDimensions: ['user'], // Only augment 'user' dimension for system.cpu
    
    // How many future data points to generate
    predictionHorizonSeconds: 300, // 5 minutes
    
    // Prediction mode: 'last-value' | 'constant' | 'linear' | 'live-fetch'
    predictionMode: 'live-fetch',
    
    // For 'constant' mode: static value
    constantValue: 50,
    
    // For 'linear' mode: slope per second
    linearSlope: 0.1,
    
    // For 'live-fetch' mode: fetch from ML endpoint
    mlEndpoint: 'http://localhost:8080/predict', // Your ML service URL
    
    // Visual marker: add dimension names for predictions
    predictionDimensionSuffix: '_predicted',
    
    // Debug logging
    debug: true
  };
  
  // ========== HELPER FUNCTIONS ==========
  
  function log(...args) {
    if (CONFIG.debug) {
      console.log('[Predictions]', ...args);
    }
  }
  
  async function generatePredictions(lastValue, count, step, mode = 'last-value', chartId = '', dimensionName = '') {
    const predictions = [];
    
    // If mode is 'live-fetch', try to fetch from ML endpoint
    if (mode === 'live-fetch' && CONFIG.mlEndpoint) {
      try {
        log(`  Fetching predictions from ML endpoint for ${dimensionName}...`);
        const response = await fetch(CONFIG.mlEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chart: chartId,
            dimension: dimensionName,
            lastValue: lastValue,
            horizonSeconds: count * step,
            points: count
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          // Expect array of numbers or { predictions: [...] }
          const preds = Array.isArray(data) ? data : (data.predictions || []);
          if (preds.length === count) {
            log(`  ✓ Received ${preds.length} predictions from ML endpoint`);
            return preds;
          }
          log(`  ⚠ ML endpoint returned ${preds.length} predictions, expected ${count}, falling back`);
        } else {
          log(`  ⚠ ML endpoint error: ${response.status}, falling back to ${mode}`);
        }
      } catch (error) {
        log(`  ⚠ Failed to fetch from ML endpoint: ${error.message}, falling back`);
      }
      // Fall back to last-value if fetch fails
      mode = 'last-value';
    }
    
    // Static prediction modes
    for (let i = 1; i <= count; i++) {
      let value;
      
      switch(mode) {
        case 'constant':
          value = CONFIG.constantValue;
          break;
          
        case 'linear':
          value = lastValue + (CONFIG.linearSlope * step * i);
          break;
          
        case 'last-value':
        default:
          value = lastValue;
          break;
      }
      
      predictions.push(value);
    }
    
    return predictions;
  }
  
  async function augmentChartData(originalData, chartId) {
    try {
      // Check if this is the target chart
      if (!chartId || !chartId.includes(CONFIG.targetChartId)) {
        return originalData;
      }
      
      log(`Augmenting chart: ${chartId}`);
      
      // Parse response (might be string or object)
      const data = typeof originalData === 'string' 
        ? JSON.parse(originalData) 
        : originalData;
      
      // Validate data structure
      if (!data || !data.result || !data.result.data) {
        log('Invalid data structure, skipping');
        return originalData;
      }
      
      const result = data.result;
      const timestamps = result.data[0]; // First row is timestamps
      const dimensionData = result.data.slice(1); // Rest are dimension values
      const dimensionNames = result.dimension_names || [];
      
      if (!timestamps || timestamps.length === 0) {
        log('No timestamps found, skipping');
        return originalData;
      }
      
      // Compute prediction parameters
      const lastTimestamp = timestamps[timestamps.length - 1];
      const step = result.view_update_every || 1; // seconds between points
      const predictionCount = Math.ceil(CONFIG.predictionHorizonSeconds / step);
      
      log(`Last timestamp: ${lastTimestamp}, step: ${step}s, adding ${predictionCount} predictions`);
      log(`Dimensions: ${dimensionNames.join(', ')}`);
      
      // Generate future timestamps
      const futureTimestamps = [];
      for (let i = 1; i <= predictionCount; i++) {
        futureTimestamps.push(lastTimestamp + (step * i));
      }
      
      // Extend timestamps array
      result.data[0] = timestamps.concat(futureTimestamps);
      
      // For each dimension, add predictions (async)
      const predictionPromises = dimensionData.map(async (dimValues, idx) => {
        const dimensionName = dimensionNames[idx] || `dim${idx}`;
        
        // Check if this dimension should be augmented
        if (CONFIG.targetDimensions.length > 0 && 
            !CONFIG.targetDimensions.includes(dimensionName)) {
          log(`  Skipping dimension: ${dimensionName} (not in targetDimensions)`);
          // For skipped dimensions, just pad with nulls
          return dimValues.concat(new Array(predictionCount).fill(null));
        }
        
        // Find last non-null value
        let lastValue = null;
        for (let i = dimValues.length - 1; i >= 0; i--) {
          if (dimValues[i] !== null && dimValues[i] !== undefined) {
            lastValue = dimValues[i];
            break;
          }
        }
        
        if (lastValue === null) {
          lastValue = 0;
        }
        
        log(`  Processing dimension: ${dimensionName}, last=${lastValue}`);
        
        // Generate predictions (async if live-fetch)
        const predictions = await generatePredictions(
          lastValue,
          predictionCount,
          step,
          CONFIG.predictionMode,
          chartId,
          dimensionName
        );
        
        log(`  Dimension ${dimensionName}: added ${predictions.length} predictions`);
        
        // Extend dimension data: keep original values, add predictions
        return dimValues.concat(predictions);
      });
      
      // Wait for all predictions to complete
      const augmentedDimensions = await Promise.all(predictionPromises);
      
      // Replace dimension data
      augmentedDimensions.forEach((augmentedData, idx) => {
        result.data[idx + 1] = augmentedData;
      });
      
      log(`Augmentation complete. New data length: ${result.data[0].length}`);
      
      return typeof originalData === 'string' 
        ? JSON.stringify(data) 
        : data;
        
    } catch (error) {
      console.error('[Predictions] Error augmenting data:', error);
      return originalData;
    }
  }
  
  // ========== INTERCEPT FETCH ==========
  
  const originalFetch = window.fetch;
  window.fetch = function(...args) {
    const url = args[0];
    
    return originalFetch.apply(this, args)
      .then(async response => {
        // Check if this is a chart data request
        if (typeof url === 'string' && 
            (url.includes('/api/v1/data') || url.includes('/api/v3/data'))) {
          
          // Extract chart ID from URL
          const urlObj = new URL(url, window.location.origin);
          const chartId = urlObj.searchParams.get('chart');
          
          // Clone response so we can read it
          const text = await response.clone().text();
          const augmentedText = await augmentChartData(text, chartId);
          
          // Return new response with augmented data
          return new Response(augmentedText, {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers
          });
        }
        
        return response;
      });
  };
  
  // ========== INTERCEPT XMLHttpRequest ==========
  
  const originalOpen = XMLHttpRequest.prototype.open;
  const originalSend = XMLHttpRequest.prototype.send;
  
  XMLHttpRequest.prototype.open = function(method, url, ...rest) {
    this._requestUrl = url;
    return originalOpen.apply(this, [method, url, ...rest]);
  };
  
  XMLHttpRequest.prototype.send = function(...args) {
    const xhr = this;
    
    // Hook onload to intercept response
    const originalOnLoad = xhr.onload;
    xhr.onload = async function() {
      if (xhr._requestUrl && 
          (xhr._requestUrl.includes('/api/v1/data') || xhr._requestUrl.includes('/api/v3/data'))) {
        
        try {
          const urlObj = new URL(xhr._requestUrl, window.location.origin);
          const chartId = urlObj.searchParams.get('chart');
          
          // Augment response (async)
          const augmented = await augmentChartData(xhr.responseText, chartId);
          
          // Override response properties
          Object.defineProperty(xhr, 'responseText', {
            writable: true,
            value: augmented
          });
          
          if (xhr.responseType === '' || xhr.responseType === 'text') {
            Object.defineProperty(xhr, 'response', {
              writable: true,
              value: augmented
            });
          } else if (xhr.responseType === 'json') {
            Object.defineProperty(xhr, 'response', {
              writable: true,
              value: typeof augmented === 'string' ? JSON.parse(augmented) : augmented
            });
          }
        } catch (error) {
          console.error('[Predictions] Error intercepting XHR:', error);
        }
      }
      
      if (originalOnLoad) {
        return originalOnLoad.apply(this, arguments);
      }
    };
    
    return originalSend.apply(this, args);
  };
  
  // ========== INITIALIZATION ==========
  
  log('Predictions injector loaded!');
  log('Configuration:', CONFIG);
  log('Intercepting chart data API calls...');
  log(`Target chart: ${CONFIG.targetChartId}`);
  log(`Target dimensions: ${CONFIG.targetDimensions.join(', ') || 'all'}`);
  log(`Prediction mode: ${CONFIG.predictionMode}`);
  if (CONFIG.predictionMode === 'live-fetch') {
    log(`ML endpoint: ${CONFIG.mlEndpoint}`);
  }
  log('Navigate to a chart or refresh the page to see predictions.');
  
  // Store original functions for cleanup
  window._netdataPredictions = {
    originalFetch,
    originalOpen,
    originalSend,
    config: CONFIG,
    
    // Method to restore original behavior
    restore: function() {
      window.fetch = originalFetch;
      XMLHttpRequest.prototype.open = originalOpen;
      XMLHttpRequest.prototype.send = originalSend;
      log('Predictions injector removed.');
    },
    
    // Method to update config
    updateConfig: function(newConfig) {
      Object.assign(CONFIG, newConfig);
      log('Configuration updated:', CONFIG);
    }
  };
  
  console.log('%c[Predictions]%c Injector active! Use window._netdataPredictions.restore() to disable.', 
    'color: #00AB44; font-weight: bold', 
    'color: inherit');
  
})();
