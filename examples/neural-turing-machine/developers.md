### Naming Convention

For parameters we should make the symbol end with `_..`, e.g,
`self.name + '_weight'`. For middle inputs, we need to use `:..`, e.g, 
`self.name + ':internal_sym'`. For sub-networks, we need to use `->..`, 
e.g., `self.name + '->net_name'`