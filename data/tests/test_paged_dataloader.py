#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import torch
from flsim.fb.data.hive_data_utils import (
    create_dataloader,
    create_paginator,
)
from flsim.fb.data.hive_dataset import InlineDatasetConfig
from flsim.fb.data.paged_dataloader import (
    PagedDataProvider,
)
from flsim.tests.utils import SampleNetHive
from libfb.py import testutil


def init_process_data_provider(rank, world_size, users_per_page, total_users):
    data_loader = create_dataloader(
        num_users_per_page=users_per_page, num_total_users=total_users, batch_size=1
    )
    data_provider = PagedDataProvider(
        data_loader,
        model=SampleNetHive(),
        page_turn_freq=0.999,
        rank=rank,
        world_size=world_size,
    )
    user_ids = []
    for user in data_provider.train_data():
        for user_examples in user:
            user_ids.append(user_examples["user_n"].item())
    return sorted(set(user_ids))


class FLPaginatorTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_page_size(self) -> None:
        hive_paginator = create_paginator(num_users_per_page=2, num_total_users=5)
        self.assertEqual(hive_paginator.page_size, 2)
        hive_paginator = create_paginator(num_users_per_page=4, num_total_users=5)
        self.assertEqual(hive_paginator.page_size, 4)

    def test_even_page_size_odd_total_users_hive(self) -> None:
        hive_paginator = create_paginator(num_users_per_page=2, num_total_users=5)
        num_pages = 0
        for _ in hive_paginator.pages():
            num_pages += 1
        self.assertEqual(num_pages, 2)

    def test_even_page_size_even_total_users_hive(self) -> None:
        hive_paginator = create_paginator(num_users_per_page=2, num_total_users=6)
        num_pages = 0
        for _ in hive_paginator.pages():
            num_pages += 1
        self.assertEqual(num_pages, 3)

    def test_odd_page_size_odd_total_users_hive(self) -> None:
        hive_paginator = create_paginator(num_users_per_page=3, num_total_users=5)
        num_pages = 0
        for _ in hive_paginator.pages():
            num_pages += 1
        self.assertEqual(num_pages, 1)

    def test_odd_page_size_even_total_users_hive(self) -> None:
        hive_paginator = create_paginator(num_users_per_page=3, num_total_users=6)
        num_pages = 0
        for _ in hive_paginator.pages():
            num_pages += 1
        self.assertEqual(num_pages, 2)

    def test_page_size_equals_total_users_hive(self) -> None:
        hive_paginator = create_paginator(num_users_per_page=5, num_total_users=5)
        num_pages = 0
        for _ in hive_paginator.pages():
            num_pages += 1
        self.assertEqual(num_pages, 1)

    def test_num_total_pages(self) -> None:
        # Same coverage as page size test above
        hive_paginator = create_paginator(num_users_per_page=2, num_total_users=5)
        self.assertEqual(hive_paginator.num_total_pages, 2)
        hive_paginator = create_paginator(num_users_per_page=4, num_total_users=5)
        self.assertEqual(hive_paginator.num_total_pages, 1)
        hive_paginator = create_paginator(num_users_per_page=2, num_total_users=6)
        self.assertEqual(hive_paginator.num_total_pages, 3)
        hive_paginator = create_paginator(num_users_per_page=3, num_total_users=5)
        self.assertEqual(hive_paginator.num_total_pages, 1)
        hive_paginator = create_paginator(num_users_per_page=3, num_total_users=6)
        self.assertEqual(hive_paginator.num_total_pages, 2)
        hive_paginator = create_paginator(num_users_per_page=5, num_total_users=5)
        self.assertEqual(hive_paginator.num_total_pages, 1)

    def test_page_sizes(self) -> None:
        page_sizes = {0: 2, 1: 2, 2: 1}
        hive_paginator = create_paginator(
            num_users_per_page=4,
            num_total_users=5,
            world_size=2,
            page_sizes=page_sizes,
        )
        self.assertEqual(hive_paginator.num_total_pages, 2)
        self.assertEqual(hive_paginator.num_total_users, 4)
        self.assertEqual(hive_paginator.cfg.num_users_per_page, 0)

        # test with default world_size=1
        hive_paginator = create_paginator(
            num_users_per_page=4,
            num_total_users=5,
            page_sizes=page_sizes,
        )
        self.assertEqual(hive_paginator.num_total_pages, 3)
        self.assertEqual(hive_paginator.num_total_users, 5)
        self.assertEqual(hive_paginator.cfg.num_users_per_page, 0)


class FLPagedDataLoaderTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def _create_inline_dataset(
        self,
        num_users=10,
        num_users_per_page=2,
        normalized_user_id=True,
        data_cols=None,
        id_col="id",
        seed=1,
    ):
        inline_config = InlineDatasetConfig(
            num_users=num_users,
            data_cols=data_cols if data_cols is not None else [],
            id_col=id_col,
        )
        return create_dataloader(
            num_users_per_page=num_users_per_page,
            num_total_users=num_users,
            batch_size=1,
            inline_config=inline_config,
            sharding_col_name=id_col,
            use_nid=normalized_user_id,
            seed=seed,
        )

    @testutil.data_provider(
        lambda: (
            {"normalized_user_id": True},
            {"normalized_user_id": False},
        )
    )
    def test_data_loader_users_per_page(self, normalized_user_id):
        num_users_per_page = 50
        dataloader = self._create_inline_dataset(
            num_users=100,
            num_users_per_page=num_users_per_page,
            normalized_user_id=normalized_user_id,
            seed=1,
        )

        pages = [list(row.keys()) for row in dataloader.fl_train_set()]
        for page in pages:
            self.assertAlmostEqual(len(page), num_users_per_page, delta=2)

    @testutil.data_provider(
        lambda: (
            {"normalized_user_id": True},
            {"normalized_user_id": False},
        )
    )
    def test_batch_structure(self, normalized_user_id) -> None:
        num_users = 50
        num_users_per_page = 10
        total_pages = num_users // num_users_per_page
        dataloader = self._create_inline_dataset(
            num_users=50,
            num_users_per_page=10,
            normalized_user_id=normalized_user_id,
            data_cols=["label", "user_n"],
            seed=1,
        )
        for users_per_round in dataloader.fl_train_set():
            for user_examples in users_per_round.values():
                for batch in user_examples:
                    self.assertTrue("label" in batch)
                    self.assertTrue("user_n" in batch)
                    self.assertTrue(isinstance(batch["label"], torch.Tensor))
                    self.assertTrue(isinstance(batch["user_n"], torch.Tensor))
        self.assertEqual(dataloader.pages_used, total_pages)


class PagedDataProviderTest(FLPagedDataLoaderTest):
    def setUp(self) -> None:
        super().setUp()

    @testutil.data_provider(lambda: ({"normalized_user_id": True},))
    def test_paged_data_provider(self, normalized_user_id):
        """
        Test basic functionality of data provider
        """
        num_total_users = 1
        dataloader = self._create_inline_dataset(
            num_users=10,
            num_users_per_page=1,
            normalized_user_id=normalized_user_id,
            data_cols=["label", "user_n"],
        )
        data_provider = PagedDataProvider(dataloader, model=SampleNetHive())
        train_data = list(data_provider.train_data())

        self.assertEqual(len(train_data), num_total_users)

    @testutil.data_provider(
        lambda: (
            {"page_turn_freq": 0.50, "is_valid": True},
            {"page_turn_freq": 0.01, "is_valid": True},
            {"page_turn_freq": 0.99, "is_valid": True},
            {"page_turn_freq": 1.0, "is_valid": False},
            {"page_turn_freq": 0, "is_valid": False},
        )
    )
    def test_page_turn_freq_validation(self, page_turn_freq, is_valid):
        """
        Param page_turn_freq should never be greater or equal to 1.0
        """
        num_users_per_page = 5
        num_total_users = 10
        data_loader = create_dataloader(
            num_users_per_page=num_users_per_page,
            num_total_users=num_total_users,
            batch_size=1,
        )

        if is_valid:
            data_provider = PagedDataProvider(
                data_loader, model=SampleNetHive(), page_turn_freq=page_turn_freq
            )
            self.assertEqual(data_provider.num_users(), num_users_per_page)
            self.assertEqual(data_provider.num_total_users(), num_total_users)
        else:
            with self.assertRaises(ValueError):
                PagedDataProvider(
                    data_loader, model=SampleNetHive(), page_turn_freq=page_turn_freq
                )

    @testutil.data_provider(
        lambda: ({"page_turn_freq": 0.99, "page_used": 5, "current_page": 0},)
    )
    def test_page_reset(self, page_turn_freq, page_used, current_page):
        """
        Test data provider loops back to the begining correctly
        """
        dataloader = self._create_inline_dataset(
            num_users=10,
            num_users_per_page=5,
            normalized_user_id=True,
            data_cols=["label", "user_n"],
        )

        data_provider = PagedDataProvider(
            dataloader, model=SampleNetHive(), page_turn_freq=page_turn_freq
        )

        page_num = []
        for i in range(23):
            user = data_provider[i]
            page_num.append(data_provider.current_page_num)
            self.assertGreaterEqual(user.num_examples(), 1)

            # page 0 expect user id between 0 and 5
            if data_provider.current_page_num == 0:
                self.assertTrue(0 < user[0]["id"].item() <= 5)
            else:
                # page 1 expect user id between 5 and 10
                self.assertTrue(5 < user[0]["id"].item() <= 10)

        self.assertEqual(
            page_num,
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        )
        self.assertEqual(data_provider.current_page_num, current_page)
        self.assertEqual(data_provider.pages_used, page_used)

    @testutil.data_provider(
        lambda: (
            {"page_turn_freq": 0.10, "page_used": 24, "current_page": 1},
            {"page_turn_freq": 0.50, "page_used": 8, "current_page": 1},
            {"page_turn_freq": 0.99, "page_used": 5, "current_page": 0},
        )
    )
    def test_page_turns_freq(self, page_turn_freq, page_used, current_page):
        """
        Test page turns correctly based on page_turn_freq
        """
        dataloader = self._create_inline_dataset(
            num_users=10,
            num_users_per_page=5,
            normalized_user_id=True,
            data_cols=["label", "user_n"],
        )

        data_provider = PagedDataProvider(
            dataloader, model=SampleNetHive(), page_turn_freq=page_turn_freq
        )

        for i in range(23):
            _ = data_provider[i]

        self.assertEqual(data_provider.current_page_num, current_page)
        self.assertEqual(data_provider.pages_used, page_used)
