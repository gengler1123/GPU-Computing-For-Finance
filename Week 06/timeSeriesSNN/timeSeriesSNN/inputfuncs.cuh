#ifndef __INPUTFUNCS_HPP__
#define __INPUTFUNCS_HPP__

#include <vector>

void func1(int t, std::vector<float> &v)
{
	int numNeurons = 200;
	if (t == 100)
	{
		for (int i = 0; i < numNeurons; i++)
		{
			v[i] = 100;
		}
		for (int i = numNeurons; i < v.size(); i++)
		{
			v[i] = 0.0f;
		}
	}
}



#endif