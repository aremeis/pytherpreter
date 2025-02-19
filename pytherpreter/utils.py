

class PrintStreamBuffer:
    def __init__(self, size: int):
        """Initialize a buffer that keeps only the last `size` characters.
        
        Args:
            size: Maximum number of characters to keep in the buffer
        """
        self.size = size
        self.buffer = ""

    def write(self, text):
        """Write text to the buffer, keeping only the last `size` characters.
        
        Args:
            text: Text to write to the buffer
        """
        if self.size > 0:
            self.buffer = (self.buffer + text)[-self.size:]
        
    def __str__(self):
        """Return the current contents of the buffer."""
        return self.buffer


