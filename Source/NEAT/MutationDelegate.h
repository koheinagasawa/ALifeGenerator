/*
* MutationDelegate.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenomeGenerator.h>

class MutationDelegate : public GenomeGenerator
{
public:
    // Structure to store information about newly added edges by mutate().
    struct MutationOut
    {
        struct NewEdgeInfo
        {
            NodeId m_sourceInNode;
            NodeId m_sourceOutNode;
            EdgeId m_newEdge;
        };

        void clear();

        static constexpr int NUM_NEW_EDGES = 3;
        NewEdgeInfo m_newEdges[NUM_NEW_EDGES];
        NodeId m_newNode = NodeId::invalid();
        int m_numNodesAdded;
        int m_numEdgesAdded;
    };

    virtual void mutate(GenomeBasePtr genomeIn, MutationOut& mutationOut) = 0;
};
