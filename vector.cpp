
#include <bits/stdc++.h>
using namespace std;
int main() {
    typedef vector< tuple<int, int, int> > my_tuple;
    my_tuple tl; 
    tl.push_back( tuple<int, int, int>(21,20,19) );
    for (my_tuple::const_iterator i = tl.begin(); i != tl.end(); ++i) {
        cout << get<0>(*i) << endl;
        cout << get<1>(*i) << endl;
        cout << get<2>(*i) << endl;
    }

    tl.push_back( tuple<int, int, int>(0,1,2) );
    cout << get<0>(tl[1]) << endl;
    cout << get<1>(tl[1]) << endl;
    cout << get<2>(tl[1]) << endl;

    return 0;
}
