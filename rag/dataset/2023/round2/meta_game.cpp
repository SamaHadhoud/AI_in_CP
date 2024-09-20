#include <deque>
#include <iostream>
#include <vector>
using namespace std;

int N;
deque<int> A, B;

bool is_metalike_now() {
  for (int i = 0; i < N; i++) {
    if (i+1 < (N+1)/2 && !(A[i] < B[i])) {
      return false;
    }
    if (i+1 > (N+1)/2 && !(A[i] > B[i])) {
      return false;
    }
    if (A[i] != B[N - i - 1]) {
      return false;
    }
  }
  return true;
}

// Simulate 2*N seconds.
// Given a vector M of O(1) candidate midpoints (index of the middle elem if N
// is odd, or right after the middle if N even), return the first time that:
// - some m in M becomes the midpoint floor(N/2) (both even and odd N), and
// - is_meta_like() holds true.
int check(vector<int> M) {
  for (int t = 0; t <= 2*N; t++) {
    for (int m : M) {
      if (m == N/2 && is_metalike_now()) {
        return t;
      }
    }
    A.push_back(B[0]);
    B.push_back(A[0]);
    A.pop_front();
    B.pop_front();
    for (int i = 0; i < (int)M.size(); i++) {
      if (--M[i] < 0) {
        M[i] += N;
      }
    }
  }
  return -1;
}


int solve() {
  cin >> N;
  A.resize(N);
  B.resize(N);
  for (int i = 0; i < N; i++) {
    cin >> A[i];
  }
  for (int i = 0; i < N; i++) {
    cin >> B[i];
  }
  // Handle any equal pairs.
  vector<int> eq;
  for (int i = 0; i < N; i++) {
    if (A[i] == B[i]) {
      if (N % 2 == 0) {
        return -1;
      }
      eq.push_back(i);
    }
  }
  if (eq.size() > 1) {
    return -1;
  }
  if (eq.size() == 1) {
    return check({eq[0]});
  }
  // From here on, guaranteed that A[i] != B[i].
  // Check the number of times that A[i] < B[i] flips.
  vector<int> flipped;
  for (int i = 1; i < N; i++) {
    if ((A[i - 1] < B[i - 1]) != (A[i] < B[i])) {
      flipped.push_back(i);
    }
  }
  if (flipped.size() > 1) {
    return -1;
  }
  // Might as well consider elements 0 and N-1 to be midpoints.
  flipped.push_back(0);
  flipped.push_back(N-1);
  return check(flipped);
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}