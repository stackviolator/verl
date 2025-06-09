def compute_score(tool_result=None, data_source=None, **kwargs):
    """
    Custom reward function for kernel benchmarking.
    
    Args:
        tool_result: JSON output from the KernelBenchTool containing execution metrics
        data_source: The source dataset identifier
        **kwargs: Additional keyword arguments
    
    Returns:
        float: The computed reward score
    """
    # If no tool result is available, return a default score
    if tool_result is None:
        print(f"Warning: No tool_result available for data_source={data_source}")
        return 0.0
    
    # Try to parse tool result if it's a string
    if isinstance(tool_result, str):
        try:
            import json
            tool_result = json.loads(tool_result)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse tool_result as JSON: {tool_result}")
            return 0.0
    
    # Extract performance metrics from tool result
    # This is a simple scoring function - you can customize based on your needs
    if isinstance(tool_result, dict):
        # Look for common performance indicators
        score = 0.0
        
        # Check if execution was successful
        if tool_result.get("success", False):
            score += 0.5  # Base score for successful execution
            
            # Check for speedup improvements
            speedup = tool_result.get("speedup", 1.0)
            if speedup > 1.0:
                score += min(0.5, (speedup - 1.0) * 0.1)  # Bonus for speedup
            
            # Check for correctness
            if tool_result.get("correctness", False):
                score += 0.3  # Bonus for correctness
                
            # Check for efficiency metrics
            memory_efficiency = tool_result.get("memory_efficiency", 1.0)
            if memory_efficiency > 1.0:
                score += min(0.2, (memory_efficiency - 1.0) * 0.1)
        
        return float(score)
    
    # Fallback scoring based on the presence of tool result
    return 0.1  # Small positive score if tool result exists but is not parseable 