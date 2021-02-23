/*
* MutationDelegate.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/MutationDelegate.h>

void MutationDelegate::MutationOut::clear()
{
    for (int i = 0; i < MAX_NUM_NEW_EDGES; i++)
    {
        m_newEdges[i].m_sourceInNode = NodeId::invalid();
        m_newEdges[i].m_sourceOutNode = NodeId::invalid();
        m_newEdges[i].m_newEdge = EdgeId::invalid();
    }

    m_numNodesAdded = 0;
    m_numEdgesAdded = 0;

    m_newNode = NodeId::invalid();
}
