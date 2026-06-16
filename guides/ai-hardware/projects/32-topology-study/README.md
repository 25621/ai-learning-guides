# Topology Study

## Key Insight

Analyzing cluster [network topology](/shared/glossary/#network-topology) reveals the physical arrangement of GPU-GPU and node-to-node connections. By running tools like [nvidia-smi](/shared/glossary/#nvidia-smi) (specifically `nvidia-smi topo -m`), developers can map the speed hierarchy of their hardware—identifying which paths utilize fast [NVLink](/shared/glossary/#nvlink) interconnects and which must route through slower [PCIe](/shared/glossary/#pcie) buses or network switches. Understanding this routing allows for the optimization of collective communication patterns, ensuring that heavy data exchanges are mapped to high-bandwidth pathways.
