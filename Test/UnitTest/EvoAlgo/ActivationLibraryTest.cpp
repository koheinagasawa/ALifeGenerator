/*
* ActivationLibraryTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <EvoAlgo/NeuralNetwork/Activations/ActivationLibrary.h>

TEST(ActivationLibrary, BasicOperations)
{
    using ActivationPtr = ActivationLibrary::ActivationPtr;

    ActivationLibrary library;

    // Empty library
    EXPECT_EQ(library.getNumActivations(), 0);
    ActivationId id0(0);
    EXPECT_FALSE(library.isActivationIdValid(id0));
    EXPECT_EQ(library.getActivationIds().size(), 0);
    EXPECT_EQ(library.getMaxActivationId(), ActivationId::invalid());
    EXPECT_EQ(library.getActivation(id0), nullptr);
    EXPECT_FALSE(library.hasActivation(nullptr));

    ActivationPtr ac0 = std::make_shared<Activation>([](float in) {return 0.f; });

    EXPECT_FALSE(library.hasActivation(ac0));

    // Register one activation
    id0 = library.registerActivation(ac0);
    ActivationId id1(1);
    EXPECT_EQ(library.getNumActivations(), 1);
    EXPECT_TRUE(library.isActivationIdValid(id0));
    EXPECT_FALSE(library.isActivationIdValid(id1));
    EXPECT_EQ(library.getActivationIds().size(), 1);
    EXPECT_EQ(library.getActivationIds()[0], id0);
    EXPECT_EQ(library.getMaxActivationId(), id0);
    EXPECT_EQ(library.getActivation(id0), ac0);
    EXPECT_TRUE(library.hasActivation(ac0));

    // Register another activation
    ActivationPtr ac1 = std::make_shared<Activation>([](float in) {return 0.f; });
    id1 = library.registerActivation(ac1);
    EXPECT_EQ(library.getNumActivations(), 2);
    EXPECT_TRUE(library.isActivationIdValid(id1));
    EXPECT_EQ(library.getActivationIds().size(), 2);
    EXPECT_EQ(library.getMaxActivationId(), id1);
    EXPECT_EQ(library.getActivation(id1), ac1);
    EXPECT_TRUE(library.hasActivation(ac1));

    // Register the activation already registered. Duplicated entry should be allowed and returned ID should be different.
    ActivationId id2 = library.registerActivation(ac1);
    EXPECT_NE(id1, id2);
    EXPECT_EQ(library.getNumActivations(), 3);
    EXPECT_EQ(library.getActivationIds().size(), 3);
    EXPECT_TRUE(library.isActivationIdValid(id2));
    EXPECT_EQ(library.getActivation(id2), ac1);

    // Unregister activation
    library.unregisterActivation(id0);
    EXPECT_EQ(library.getNumActivations(), 2);
    EXPECT_FALSE(library.isActivationIdValid(id0));
    EXPECT_TRUE(library.isActivationIdValid(id1));
    EXPECT_TRUE(library.isActivationIdValid(id2));
    EXPECT_EQ(library.getActivationIds().size(), 2);
    EXPECT_EQ(library.getMaxActivationId(), id2);
    EXPECT_EQ(library.getActivation(id0), nullptr);
    EXPECT_EQ(library.getActivation(id1), ac1);
    EXPECT_FALSE(library.hasActivation(ac0));
    EXPECT_TRUE(library.hasActivation(ac1));

    // Try to register nullptr
    ActivationId id3 = library.registerActivation(nullptr);
    EXPECT_EQ(id3, ActivationId::invalid());
    EXPECT_EQ(library.getNumActivations(), 2);

    // Try to register unregistered ID.
    library.unregisterActivation(ActivationId(100));
    EXPECT_EQ(library.getNumActivations(), 2);
}
