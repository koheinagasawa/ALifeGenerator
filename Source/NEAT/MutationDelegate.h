/*
* MutationDelegate.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenomeGenerator.h>

// GenomeGenerator which creates a new genome by mutating existing one.
class MutationDelegate : public GenomeGenerator
{
public:
    // Struct to store information about newly added node and edges by mutate().
    struct MutationOut
    {
        struct NewEdgeInfo
        {
            NodeId m_sourceInNode;
            NodeId m_sourceOutNode;
            EdgeId m_newEdge;
        };

        struct NewNodeInfo
        {
            NodeId m_newNode;
            EdgeId m_previousEdgeId;
            EdgeId m_newIncomingEdgeId;
            EdgeId m_newOutgoingEdgeId;
        };

        void clear();

        static constexpr int MAX_NUM_NEW_EDGES = 3;
        NewEdgeInfo m_newEdges[MAX_NUM_NEW_EDGES];  // Info of newly added edges.
        NewNodeInfo m_newNode;                      // Info of newly added node.
        int m_numNodesAdded;
        int m_numEdgesAdded;
    };

    // Mutate a single genome. There are three ways of mutation.
    virtual void mutate(GenomeBase* genomeInOut, MutationOut& mutationOut) = 0;
};
