class Stack:
    """A stack is a data structure that allows for the last element added to be the first element removed."""
    def __init__(self):
        self.stack = []

    def is_empty(self):
        return self.size() == 0

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if self.is_empty():
            return None
        return self.stack.pop()
    
    def peek(self):
        if self.is_empty():
            return None
        return self.stack[-1]
    
    def size(self):
        return len(self.stack)

    def __str__(self):
        return str(self.stack)

class Queue:
    """A queue is a data structure that allows for the first element added to be the first element removed."""
    def __init__(self):
        self.queue = []

    def is_empty(self):
        return self.size() == 0 

    def enqueue(self, value):
        self.queue.append(value)

    def dequeue(self):
        if self.size() == 0:
            return None
        return self.queue.pop(0)
    
    def size(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)

def test_stack_push():
    my_stack = Stack()
    assert my_stack.size() == 0
    my_stack.push(1)
    assert my_stack.size() == 1
    assert my_stack.peek() == 1
    my_stack.push(2)
    my_stack.push(3)
    assert my_stack.peek() == 3
    assert my_stack.size() == 3
    print("All stack push tests pass")

def test_stack_pop():
    my_stack = Stack()
    my_stack.push(1)
    my_stack.push(2)
    assert my_stack.pop() == 2
    assert my_stack.size() == 1
    assert my_stack.pop() == 1
    assert my_stack.size() == 0
    assert my_stack.pop() == None
    print("All stack pop tests pass")

def test_queue_enqueue():
    my_queue = Queue()
    assert my_queue.size() == 0
    my_queue.enqueue(1)
    assert my_queue.size() == 1
    my_queue.enqueue(2)
    my_queue.enqueue(3)
    assert my_queue.size() == 3
    print("All queue enqueue tests pass")

def test_queue_dequeue():
    my_queue = Queue()
    my_queue.enqueue(1)
    my_queue.enqueue(2)
    assert my_queue.dequeue() == 1
    assert my_queue.size() == 1
    assert my_queue.dequeue() == 2
    assert my_queue.size() == 0
    assert my_queue.dequeue() == None
    print("All queue dequeue tests pass")

def main():
    test_stack_push()
    test_stack_pop()
    test_queue_enqueue()
    test_queue_dequeue()

if __name__ == "__main__":
    main()  
