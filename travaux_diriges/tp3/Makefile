CXX = mpic++
CXXFLAGS = -O2 -std=c++17
TARGET = bucket_sort
SRC = Bucket_Sort.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
