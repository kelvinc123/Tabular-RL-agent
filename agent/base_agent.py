from collections import defaultdict
import numpy as np

class Agent:
    '''
    Parent class for agents. All agents must inherit Agent class.
    '''
    
    def __init__(self):
        '''
        Initialize:
            qvalues : q values of all pair of states and actions
            learn : indicates if the agent is learning or not
        '''
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self._learn = True

    def _dict_to_defaultdict(self, dictionary):
        '''
        Helper method to update dictionary to defaultdict
        Input:
            dictionary: dict : dictionary
        Output:
            res: defaultdict : defaultdict which has the same content as dictionary
        '''
        res = defaultdict(lambda: 0)
        res.update(dictionary)
        return res
        
    def learning_mode_on(self):
        '''
        Method to turn on learning mode
        '''
        self._learn = True
        
    def learning_mode_off(self):
        '''
        Method to turn off learning mode
        '''
        self._learn = False
        
    def get_qvalue(self, state, action):
        '''
        Method to get q value given state and action
        '''
        state=tuple(list(state))
        action=tuple(list(action))
        return self._qvalues[state][action]
    
    def set_qvalue(self, state, action, value):
        '''
        Function to set q value given state and action
        '''
        state=tuple(list(state))
        action=tuple(list(action))
        self._qvalues[state][action] = value
    
    def get_value(self, state):
        '''
        Function to get v value from a given state.
        Must be implemented on the child class
        '''
        raise NotImplementedError
        
    def update(self, state, action, reward, next_state):
        
        '''
        Method to learn.
        Must be implemented on the child class
        '''
        raise NotImplementedError

    def save_model(self, filename):
        '''
        Method to save the learned q value to a file
        Input:
            filename: string : filename for model, ends with .npy
        '''

        # Convert defaultdict of _qvalues to dictionary
        dictionary_q = {
            key: {k: v for k, v in val.items()} for key, val in self._qvalues.items()
        }

        # Save the dictionary
        np.save(filename, dictionary_q)

    def load_model(self, filename):
        '''
        Method to load q value from a file
        Input:
            filename: string : filename for model, ends with .npy
        '''

        # Load the dictionary
        dict_q = np.load(filename, allow_pickle=True).item()

        # Change the dictionary type of actions to defaultdict
        dict_q = {key: self._dict_to_defaultdict(val) for key, val in dict_q.items()}

        # Reinitialize q values
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))

        # Update q values using dict_q
        self._qvalues.update(dict_q)        

class RandomAgent(Agent):
    
    '''
    Random agent that behaves randomly.
    '''
    
    def __init__(self):
        super().__init__()
        
    def get_value(self, state, possible_actions):
        '''
        Method to get value on a given state

        Input:
            state: str
                current state
            possible_actions: list
                list of all possible action given the state
        '''
        # Get all possible actions
        actions = possible_actions.copy()

        # Get all q value of all state and actions
        q_val = [self.get_qvalue(state, action) for action in actions]

        if len(q_val) == 0:
            return 0

        # Expected q value of taking action randomly
        return np.sum(q_val) / len(q_val)
        
    def get_action(self, possible_actions):
        
        '''
        Method to get action given state

        Input:
            possible_actions: list
                list of all possible action
        '''
        
        # Get all possible actions
        actions = possible_actions.copy()
        
        # Return none if no valid action
        if len(actions) == 0:
            return None
        
        # Pick an action randomly
        idx_choice = np.random.choice(range(len(actions)))
        
        return actions[idx_choice]
    
    def update(self, state, action, reward, next_state):
        pass