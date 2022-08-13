CC=g++
CFLAGS=


SOURCES=model/matrix/*.cpp model/MLP/*.cpp model/*.cpp
TEST_SOURCES=tests.cpp


main:
	$(CC) $(SOURCES) main.cpp -o main.out

tests:
	$(CC) $(SOURCES) $(TEST_SOURCES) -lgtest -o tests.out
	./tests.out

clean:
	rm -rf *.out *.o
	
.PHONY: main