import sys
import time
from datetime import timedelta

class Progress:
    def __init__(self,
                 iterable,
                 desc     = '',
                 total    = None,
                 miniters = 0.1,
                 file     = sys.stdout,
                 ):
        self.iterable = iterable
        self.desc = desc
        if total is None:
            self.total = len(iterable)
        else:
            self.total = total
        
        self.items_field_width = len(str(self.total))
        
        if isinstance(miniters, int):
            self.miniters = miniters
        else:
            self.miniters = int(round(miniters * self.total))
        
        self.file = file
        
        self.current = 0
        self.start_time = time.time()
    
    def set_description(self, desc):
        self.desc = desc
    
    def update(self, n=1):
        self.current += n
    
    def show(self):
        if (not self.current % self.miniters) or (self.current == self.total-1):
            items = self.current + 1
            perc = int(round(items/self.total*100))
            elapsed = time.time() - self.start_time
            items_per_sec = items/(elapsed if elapsed else 1)
            projected_total = elapsed * self.total/items
            
            elapsed_str = timedelta(seconds=int(elapsed))
            projected_str = timedelta(seconds=int(projected_total))
            
            print(f'{self.desc}: '
                  f'{perc:3d}%| {items:{self.items_field_width}d}/{self.total:{self.items_field_width}d} | '
                  f'[{elapsed_str}<{projected_str}] '
                  f'{items_per_sec:.2f}it/s',
                  file=self.file, flush=True)
    
    def __iter__(self):
        self.start_time = time.time()
        for i, item in enumerate(self.iterable):
            self.current = i
            self.show()
            yield item
    
    def __enter__(self):
        self.current = 0
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.close()
    
    def close(self):
        self.current = self.total-1
        self.file.flush()

