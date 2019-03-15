### Coding style
When writing an analyzer, don't use magic numbers. Instead you can implement them as config variables or constants on module level.
#### BAD:
```python
class TrivialAnalyzer(CHIMEAnalyzer):
    def run(self):
        freqs = [1, 2, 3]
        for f in range freqs:
            if f > 2:
```

#### GOOD:
```python
CHIME_N2_FREQS = [1, 2, 3]

class TrivialAnalyzer(CHIMEAnalyzer):
    max_freq = Property(proptype=int)
    
    def run(self):
        for f in CHIME_N2_FREQS:
            if f > self.max_freq:
