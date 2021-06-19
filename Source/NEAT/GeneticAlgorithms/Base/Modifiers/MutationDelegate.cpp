/*
* MutationDelegate.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/Base/Modifiers/MutationDelegate.h>

void MutationDelegate::MutationOut::clear()
{
    for (int i = 0; i < MAX_NUM_NEW_EDGES; i++)
    {
        m_newEdgeInfos[i].m_sourceInNode = NodeId::invalid();
        m_newEdgeInfos[i].m_sourceOutNode = NodeId::invalid();
        m_newEdgeInfos[i].m_edgeId = EdgeId::invalid();
    }

    m_newNodeInfo.m_nodeId = NodeId::invalid();
    m_newNodeInfo.m_previousEdgeId = EdgeId::invalid();
    m_newNodeInfo.m_newIncomingEdgeId = EdgeId::invalid();
    m_newNodeInfo.m_newOutgoingEdgeId = EdgeId::invalid();

    m_numEdgesAdded = 0;
}
