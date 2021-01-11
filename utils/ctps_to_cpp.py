def main():
    exponents = [(0, 1, 1),
                 (0, 1, 2),
                 (1, 4, 0),
                 (1, 3, 0),
                 (1, 4, 1),
                 (1, 3, 1),
                 (1, 4, 2),
                 (1, 3, 2),
                 (1, 2, 0),
                 (1, 2, 1),
                 (1, 2, 2),
                 (2, 3, 0),
                 (3, 4, 0),
                 (2, 3, 1),
                 (3, 4, 1),
                 (4, 5, 0),
                 (2, 3, 2),
                 (3, 4, 2),
                 (1, 1, 0),
                 (1, 1, 1),
                 (1, 1, 2),
                 (5, 4, 0),
                 (5, 4, 1),
                 (4, 3, 0),
                 (4, 3, 1),
                 (3, 2, 0),
                 (3, 2, 1),
                 (3, 2, 2),
                 (5, 3, 0),
                 (7, 4, 0),
                 (2, 1, 0),
                 (2, 1, 1),
                 (2, 1, 2),
                 (9, 4, 0),
                 (7, 3, 0),
                 (5, 2, 0),
                 (5, 2, 1),
                 (5, 2, 2),
                 (8, 3, 0),
                 (11, 4, 0),
                 (3, 1, 0),
                 (3, 1, 1)]
    for i, j, k in exponents:
        if i // j == i / j:
            print(f"{i//j}, {k},")
        else:
            print(f"{i}./{j}, {k},")


if __name__ == '__main__':
    main()
