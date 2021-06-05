############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections
import random
import time


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 1000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # The action in discrete form
        self.discrete_action = None
        # Create agent's total reward for current episode
        self.total_reward = None   # look into this as never reset the agent
        # Set the agent's epsilon value
        self.epsilon = 0.8


        # Set episode counter
        self.nr_episodes = 0
        # Create a DQN (Deep Q-Network)
        self.dqn = DQN()
        # Create an empty replay buffer
        self.replay_buffer = ReplayBuffer()
        # Define size of minibatch
        self.minibatch_size = 1849
        # Check to see if the policy just previously run is greedy
        self.greedy_policy = False
        # Check to see if the training should end
        self.finished_training = False
        # Set the start time of the training
        self.start_time = time.time()


    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):

        if self.num_steps_taken % self.episode_length == 0:
            # Reset the agent's reward to zero at the end of an episode
            self.total_reward = 0
            self.nr_episodes += 1
            # Reset number of steps
            self.num_steps_taken = 0
            return True
        else:
            return False


    def get_next_action(self, state):
        # Choose next action
        if self.nr_episodes < 5:
            self.discrete_action = self._choose_next_action(self.dqn, state)
            # Convert the discrete action into a continuous action.
            action = self._discrete_action_to_continuous(self.discrete_action)

        elif self.nr_episodes < 10:

            # epsilon decay
            self.epsilon = self.epsilon * 0.9999
            self.discrete_action = self._choose_next_action(self.dqn, state)
            # Convert the discrete action into a continuous action.
            action = self._discrete_action_to_continuous(self.discrete_action)

        else:
            if time.time() < self.start_time + 420:
                self.episode_length = 420
                if (self.nr_episodes % 3 == 0 or self.nr_episodes % 3 == 1) and not self.finished_training:
                    self.greedy_policy = False

                    if self.num_steps_taken < 80:

                        self.discrete_action = self.get_greedy_action_discrete(state)
                        action = self._discrete_action_to_continuous(self.discrete_action)


                    else:
                        self.epsilon = 0.8
                        self.discrete_action = self._choose_next_action(self.dqn, state)
                        action = self._discrete_action_to_continuous(self.discrete_action)
                        self.greedy_policy = False

                else:
                    self.episode_length = 100
                    self.discrete_action = self.get_greedy_action_discrete(state)
                    self.greedy_policy = True
                    action = self._discrete_action_to_continuous(self.discrete_action)


            else:

                self.episode_length = 280
                if (self.nr_episodes % 3 == 0 or self.nr_episodes % 3 == 1) and not self.finished_training:
                    self.greedy_policy = False

                    if self.num_steps_taken < 80:

                        self.discrete_action = self.get_greedy_action_discrete(state)
                        action = self._discrete_action_to_continuous(self.discrete_action)



                    else:
                        self.epsilon = 0.8
                        self.discrete_action = self._choose_next_action(self.dqn, state)
                        action = self._discrete_action_to_continuous(self.discrete_action)
                        self.greedy_policy = False

                else:
                    self.episode_length = 100
                    self.discrete_action = self.get_greedy_action_discrete(state)
                    self.greedy_policy = True
                    action = self._discrete_action_to_continuous(self.discrete_action)



        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        # Must return a continuous action
        return action


    """Function to set the next state and distance, which resulted from applying 
    action self.action at state self.state"""
    def set_next_state_and_distance(self, next_state, distance_to_goal):

        if time.time() > self.start_time + 30:
            self.end_the_search(distance_to_goal, 0.03)

        if time.time() > self.start_time + 450:

            self.end_the_search(distance_to_goal, 0.1)

        if time.time() > self.start_time + 540:

            self.end_the_search(distance_to_goal, 0.25)

        if time.time() > self.start_time + 570:

            self.end_the_search(distance_to_goal, 0.5)






        if not self.finished_training and not self.greedy_policy:
            # Convert the distance to a reward
            reward = float(0.4*((1 - distance_to_goal)**2))
            # Create a transition
            transition = (self.state, self.discrete_action, reward, next_state)

            # Update the agent's reward for this episode
            self.total_reward += reward

            self.replay_buffer.add_transition(transition)


        if not self.finished_training and not self.greedy_policy:

            if len(self.replay_buffer.buffer) > (self.minibatch_size):
                minibatch = self.replay_buffer.sample_minibatch(self.minibatch_size)
                self.dqn.train_q_network(minibatch)

                # Every N (here let N=5) episodes we update the target network
                if self.nr_episodes % 5 == 0:
                    self.dqn.update_target_network()



    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        state = torch.tensor(state)
        predicted_q_for_all_this_state = self.dqn.q_network.forward(state)
        max_Q_value_action = torch.argmax(predicted_q_for_all_this_state)
        action = self._discrete_action_to_continuous(max_Q_value_action)

        return action


    def get_greedy_action_discrete(self, state):
        state = torch.tensor(state)
        predicted_q_for_all_this_state = self.dqn.q_network.forward(state)
        action = torch.argmax(predicted_q_for_all_this_state)

        return action


    def end_the_search(self, distance_to_goal, end_paramter):
        if self.num_steps_taken <= 100 and distance_to_goal < end_paramter and self.greedy_policy == True:
            self.finished_training = True





    # Function for the agent to choose its next action
    def _choose_next_action(self, Network, input_state):
        # Implement epsilon greedy policy
        list_of_actions = [0, 1, 2]
        # Pick a random number between 0 and 1
        random_betw_0_1 = random.random()

        # Implement epsilon greedy policy
        if random_betw_0_1 < self.epsilon:
            return np.random.choice(list_of_actions)
        else:
            state = torch.tensor(input_state)
            all_Q_values = Network.q_network.forward(state)
            max_Q_value_action = torch.argmax(all_Q_values).item()

            return max_Q_value_action


    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)

        if discrete_action == 1:
            # Move 0.1 to the upwards, and 0 right
            continuous_action = np.array([0, 0.02], dtype=np.float32)

        if discrete_action == 2:
            # Move 0.1 to the downwards, and 0 right
            continuous_action = np.array([0, -0.02], dtype=np.float32)

        return continuous_action




# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This network has four hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_3 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_4 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output.
    # In this example, a ReLU activation function is used for all hidden layers, but the output
    # layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))

        output = self.output_layer(layer_4_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)

        # Create a target Q-network to mirror the original Q-network
        self.target_q_network = Network(input_dimension=2, output_dimension=3)

        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    def update_target_network(self):
        # Copy weights from Q-network into the target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())



    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss, predicted_q_for_all, minib_states_tensor = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()

        # Reshaping the q-value tensor with all the 4 actions (can relate to the
        # minib_states_tensor in loss function to know which state each index is
        reshaped_tensor = self.calculate_dict_for_10x10x3_tensor(predicted_q_for_all)
        # Return the loss as a scalar
        return loss.item(), reshaped_tensor


    # turning the 100x4 tensor into a 10x10x4 tensor
    # where the row [11] initially becomes [1][0] in the new tensor
    def calculate_dict_for_10x10x3_tensor(self, predicted_q_for_all):
        reshaped_tensor = torch.reshape(predicted_q_for_all, (43,43,3))
        return reshaped_tensor



    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):

        # We have the reward from the tuple which is the only reward we want for now
        # Use MSE between network prediction for SA and actual reward for SA
        minib_states_tensor = torch.tensor(transition[0])

        minib_actions_tensor = torch.tensor(transition[1])
        # R
        minib_rewards_tensor = torch.tensor(transition[2], dtype = torch.float32)
        # Next states
        minib_next_states_tensor = torch.tensor(transition[3])

        # Q(S, a) for all actions for this state
        predicted_q_for_all_this_state = self.q_network.forward(minib_states_tensor)

        # Q(S,A)
        predicted_q_value_tensor_this_state = predicted_q_for_all_this_state.gather(dim=1, index = minib_actions_tensor.unsqueeze(-1)).squeeze(-1)

        # Q(S',a) for all actions for next state derived from the target Q network
        predicted_q_for_all_next_state = self.target_q_network.forward(minib_next_states_tensor).detach()

        max_Q_value_action = torch.argmax(predicted_q_for_all_next_state, dim=1)

        # Double Q network
        double_deep_Q_ = self.q_network.forward(minib_next_states_tensor).gather(dim=1, index= max_Q_value_action.unsqueeze(-1)).squeeze(-1)

        gamma_multiplication = torch.mul(double_deep_Q_, 0.95)

        # R + max(over a) Q(S',a)
        expected_discounted_sum_f_rewards = gamma_multiplication + minib_rewards_tensor

        # P 35 from the lecture 1 using that equation for the loss function
        loss = torch.nn.MSELoss()(predicted_q_value_tensor_this_state, expected_discounted_sum_f_rewards)

        loss = loss.to(dtype=torch.float32)

        return loss, predicted_q_for_all_this_state, minib_states_tensor


class ReplayBuffer:

    def __init__(self):
        # Dynamically create large dataset of data tuples
        self.buffer = collections.deque(maxlen=20000)

    def add_transition(self, transition):
        self.buffer.append(transition)

    def sample_minibatch(self, minibatch_size):
        # Have to sample a random minibatch of transitions from replay buffer
        sample_state = []
        sample_action = []
        sample_reward = []
        sample_next_state = []

        indexes_to_choose = np.random.choice(len(self.buffer), minibatch_size, replace= False)
        for index in indexes_to_choose:
            sample_state.append(self.buffer[index][0])
            sample_action.append(self.buffer[index][1])
            sample_reward.append(self.buffer[index][2])
            sample_next_state.append(self.buffer[index][3])

        result = (sample_state, sample_action, sample_reward, sample_next_state)
        return result







