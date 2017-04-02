### Naming Convention

For inputs, we need to use `:..`, e.g, `self.name + ':fc1_weight'`, 
`self.name + ':internal_sym'`. For sub-networks, we need to use `->..`, 
e.g., `self.name + '->net_name'`