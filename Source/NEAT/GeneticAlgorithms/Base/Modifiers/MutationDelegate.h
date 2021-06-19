/*
* MutationDelegate.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/Base/Modifiers/GenomeModifier.h>

// GenomeModifier which mutates genomes.
class MutationDelegate : public GenomeModifier
{
public:
    // Struct to store information about newly added node and edges by mutate().
    struct MutationOut
    {
        struct NewEdgeInfo
        {
            EdgeId m_edgeId;
            NodeId m_sourceInNode;
            NodeId m_sourceOutNode;
        };

        struct NewNodeInfo
        {
            NodeId m_nodeId;
            EdgeId m_previousEdgeId;
            EdgeId m_newIncomingEdgeId;
            EdgeId m_newOutgoingEdgeId;
        };

        void clear();

        static constexpr int MAX_NUM_NEW_EDGES = 3;
        NewEdgeInfo m_newEdgeInfos[MAX_NUM_NEW_EDGES];  // Info of newly added edges.
        NewNodeInfo m_newNodeInfo;                      // Info of newly added node.
        int m_numEdgesAdded;                            // The number of newly added edges.
    };

    // Mutate a single genome. There are three ways of mutation.
    virtual void mutate(GenomeBase* genomeInOut, MutationOut& mutationOut) = 0;
};
