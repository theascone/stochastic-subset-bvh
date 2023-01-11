#include "core/counters.h"

int& getNodeIntersectionsCounter() {
    thread_local int counter;
    return counter;
}

int& getPrimIntersectionsCounter() {
    thread_local int counter;
    return counter;
}