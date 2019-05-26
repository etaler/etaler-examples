#define CNN_SINGLE_THREAD
#define CNN_USE_AVX
#include <cstdlib>
#include <iostream>
#include <vector>
#include <iomanip>

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/xtensor/xadapt.hpp"
#include "tiny_dnn/xtensor/xio.hpp"
#include "tiny_dnn/xtensor/xsort.hpp"
#include "tiny_dnn/xtensor/xnpy.hpp"

#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Encoders/Category.hpp>
#include <Etaler/XtensorInterop.hpp>

//parameters for RNNPlayer
const int RNN_DATA_PER_EPOCH = 3;

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

template <typename N>
void constructNet(N &nn, const std::string rnn_type) {
	using recurrent = tiny_dnn::recurrent_layer;
	const int hidden_size = 10; // recurrent state size
	const int seq_len = RNN_DATA_PER_EPOCH; // input sequence length

	if (rnn_type == "rnn") {
		nn << recurrent(rnn(3, hidden_size), seq_len);
	} else if (rnn_type == "gru") {
		nn << recurrent(gru(3, hidden_size), seq_len);
	} else if (rnn_type == "lstm") {
		nn << recurrent(lstm(3, hidden_size), seq_len);
	}
	nn << leaky_relu() << fc(hidden_size, 3) << softmax();
	
	nn.template at<recurrent_layer>(0).bptt_max(RNN_DATA_PER_EPOCH);
}

struct RNNPlayer
{
public:
	RNNPlayer()
	{
		constructNet(nn_, "gru");
		nn_.set_netphase(net_phase::test);
		nn_.at<recurrent_layer>(0).seq_len(1);
	}
	
	xt::xarray<float> compute(int last_oppo_move)
	{
		xt::xarray<float> in = xt::zeros<float>({3});
		in[last_oppo_move%3] = 1;
		return compute(in);
	}
	
	xt::xarray<float> compute(xt::xarray<float> input)
	{
		assert(input.size() == 3);
		//save data for traning
		if(last_input_.size() != 0) {
			for(auto v : last_input_)
				input_.push_back(v);
			for(auto v : input)
				output_.push_back(v);
		}
		last_input_ = vec_t(input.begin(), input.end());
		
		//Train once all needed data collected
		if(input_.size() == RNN_DATA_PER_EPOCH) {
			assert(input_.size() == output_.size());
			nn_.at<recurrent_layer>(0).seq_len(RNN_DATA_PER_EPOCH);
			nn_.set_netphase(net_phase::train);
			nn_.fit<cross_entropy_multiclass>(optimizer_, std::vector<vec_t>({input_}),std::vector<vec_t>({output_}), 1, 1, [](){},[](){});
			nn_.set_netphase(net_phase::test);
			nn_.at<recurrent_layer>(0).seq_len(1);
			
			input_.clear();
			output_.clear();
		}
		
		
		//Predict the opponent's next mvoe
		vec_t out = nn_.predict(vec_t(input.begin(), input.end()));
		
		assert(out.size() == 3);
		//Convert to xarray
		xt::xarray<float> r = xt::zeros<float>({3});
		for(size_t i=0;i<out.size();i++)
			r[i] = out[i];
		
		return r;
		
	}
	
	vec_t input_;
	vec_t output_;
	network<sequential> nn_;
	nesterov_momentum optimizer_;
	vec_t last_input_;
};

//parameters for HTMPlayer
const int TM_DEPTH = 32;
const int ENCODE_WIDTH = 24;

using namespace et;

struct HTMPlayer
{
public:
	HTMPlayer() :
		tm_({3* ENCODE_WIDTH}, TM_DEPTH, 1024)
	{
		last_state_ = zeros({3*ENCODE_WIDTH, TM_DEPTH}, DType::Bool);
		tm_.setPermanceDec(0.095);
		tm_.setPermanceInc(0.045);
		tm_.setActiveThreshold(4);
	}
	
	xt::xarray<float> compute(int last_oppo_move, bool learn = true)
	{
		auto[predict, active] = tm_.compute(encoder::category(last_oppo_move, 3, ENCODE_WIDTH), last_state_);

		if(learn)
			tm_.learn(active, last_state_);

		last_state_ = active;

		et::Tensor predicted_sdr = sum(predict, 1);
		et::Tensor prediction_strength = predicted_sdr.reshape({3, ENCODE_WIDTH}).sum(1).cast(DType::Float);
		
		return to_xarray<float>(prediction_strength);
	}
	
	TemporalMemory tm_;
	et::Tensor last_state_;
};


enum Move
{
	Rock,
	Paper,
	Scissor
};

//Converts agent predictions to agent move
int predToMove(int pred)
{
	if(pred == Rock)
		return Paper;
	else if(pred == Paper)
		return Scissor;
	else
		return Rock;
}

//1 - first agent wins
//0 - draw
//-1 -second agent winds
int winner(int move1, int move2)
{
	if(move1 == move2)
		return 0;
	if(move1 == Rock && move2 == Paper)
		return -1;
	
	if(move1 == Paper && move2 == Scissor)
		return -1;
	
	if(move1 == Scissor && move2 == Rock)
		return -1;
	
	return 1;
}

std::string move2String(int move)
{
	if(move == Rock)
		return "Rock";
	else if(move == Paper)
		return "Paper";
	return "Scissor";
}

//xtensor does not provide a softmax function.
xt::xarray<float> softmax(const xt::xarray<float>& x)
{
	auto b = xt::eval(xt::exp(x-xt::amax(x)[0]));
	return b/xt::sum(b)[0];
}

//A hacky argmax implementation
template <typename T>
size_t argmax(const xt::xarray<T>& arr)
{
	T max_val = arr[0];
	size_t idx = 0;
	
	for(size_t i=0;i<arr.size();i++) {
		if(max_val < arr[i]) {
			idx = i;
			max_val = arr[i];
		}
	}
	return idx;
}

int main(int argc, char** argv)
{
	bool verbose = false;
	if(argc > 1) {
		std::string opt = argv[1];
		if(opt == "-v" || opt == "--verbose") {
			verbose = true;

		}
		else {
			std::cout << "Usage: rock_paper_scissors [-h|--help] [-v|--verbose]\n\t-h help\n\t-a auto play" << std::endl;
			return 0;
		}
	}

	//OpenCL is slower in such a small network
	//setDefaultBackend(std::make_shared<OpenCLBackend>());

	//Initialize both AI
	RNNPlayer player1;
	HTMPlayer player2;
	
	int rnn_last_move = 0;
	int htm_last_move = 0;

	size_t rnn_win = 0;
	size_t draw = 0;
	size_t htm_win = 0;
	

	int num_games = 20*10000;
	//xt::xarray<int> results = xt::zeros<int>({num_games});
	for(int i=0;i<num_games;i++) {
		//Run RNN
		auto rnn_out = player1.compute(htm_last_move);
		int rnn_pred = argmax(rnn_out);
		
		//Run HTM
		auto htm_out = player2.compute(rnn_last_move);
		int htm_pred = argmax(htm_out);
		
		int rnn_move = predToMove(rnn_pred);
		int htm_move = predToMove(htm_pred);
		
		int winner_algo = winner(rnn_move, htm_move);

		if(verbose) {
			std::cout << "Round " << i << std::endl;
			std::cout << "RNN pred: " << rnn_out << ", HTM pred: " << ::softmax(htm_out) << std::endl;
			std::cout << "RNN: " << move2String(rnn_move) << ", " << "HTM: " << move2String(htm_move)
				<< ", Winner: "<< (winner_algo==1?"\033[1;31mRNN":(winner_algo==0?"draw":"\033[1;32mHTM")) << "\033[1;0m" << std::endl;
			std::cout << std::endl;
		}

		
		rnn_last_move = rnn_move;
		htm_last_move = htm_move;
		
		if(winner_algo == 1)
			rnn_win += 1;
		else if(winner_algo == 0)
			draw += 1;
		else
			htm_win += 1;
			
		if(i%1000 == 0)
			std::cout << i << '\r' << std::flush;
		//results[i] = winner_algo;
	}

	std::cout << std::fixed;

	std::cout << "After all the battles" << std::endl;
	std::cout << "RNN Wins " << rnn_win << " times, " << (float)rnn_win/num_games*100 << "%\n";
	std::cout << "HTM Wins " << htm_win << " times, " << (float)htm_win/num_games*100 << "%\n";
	std::cout << "draw: " << draw << std::endl;
	//xt::dump_npy("results.npy", results);


}
