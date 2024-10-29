#include<iostream>
#include <limits>
using namespace std;

int main() {

  int c = 0;
  float fl = 1;
  while(fl != numeric_limits<float>::infinity()) {
    fl *= 2;
    c += 1;
  }
  cout << "Infinity for float is 2^" << c << endl;

  c = 0;
  fl = 1;
  while(fl != 0) {
    fl /= 2;
    c += 1;
  }
  cout << "Zero for float is 2^-" << c << endl;

  c = 0;
  fl = 1;
  while(fl + 1 > 1) {
    fl /= 2;
    c += 1;
  }
  cout << "Epsilon for float is 2^-" << c << endl << endl;

  double db = 1;
  c = 0;
  while(db != numeric_limits<double>::infinity()) {
    db *= 2;
    c += 1;
  }
  cout << "Infinity for double is 2^" << c << endl;

  c = 0;
  db = 1;
  while(db != 0) {
    db /= 2;
    c += 1;
  }
  cout << "Zero for double is 2^-" << c << endl;

  c = 0;
  db = 1;
  while(db + 1 > 1) {
    db /= 2;
    c += 1;
  }
  cout << "Epsilon for double is 2^-" << c << endl << endl;

  long double ldb = 1;
  c = 0;
  while(ldb != numeric_limits<long double>::infinity()) {
    ldb *= 2;
    c += 1;
  }
  cout << "Infinity for long double is 2^" << c << endl;

  c = 0;
  ldb = 1;
  while(ldb != 0) {
    ldb /= 2;
    c += 1;
  }
  cout << "Zero for long double is 2^-" << c << endl;

  c = 0;
  ldb = 1;
  while(ldb + 1 > 1) {
    ldb /= 2;
    c += 1;
  }
  cout << "Epsilon for long double is 2^-" << c << endl << endl;

  return 0;
}
