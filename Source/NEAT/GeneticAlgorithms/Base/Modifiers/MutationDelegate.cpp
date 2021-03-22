/*
* MutationDelegate.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/Base/Modifiers/MutationDelegate.h>

void MutationDelegate::MutationOut::clear()
{
    for (int i = 0; i < MAX_NUM_NEW_EDGES; i++)
    {
        m_newEdges[i].m_sourceInNode = NodeId::invalid();
        m_newEdges[i].m_sourceOutNode = NodeId::invalid();
        m_newEdges[i].m_newEdge = EdgeId::invalid();
    }

    m_newNode.m_newNode = NodeId::invalid();
    m_newNode.m_previousEdgeId = EdgeId::invalid();
    m_newNode.m_newIncomingEdgeId = EdgeId::invalid();
    m_newNode.m_newOutgoingEdgeId = EdgeId::invalid();

    m_numNodesAdded = 0;
    m_numEdgesAdded = 0;
}
