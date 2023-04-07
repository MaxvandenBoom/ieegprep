"""
Functions to format and color console text

=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging


class ConsoleColors:
    """
    Class to hold console color code constants
    """

    RESET = '\033[0m'
    BOLD = '\033[1m'
    DISABLE = '\033[2m'
    UNDERLINE = '\033[4m'
    REVERSE = '\033[7m'
    INVISIBLE = '\033[8m'
    STRIKETHROUGH = '\033[9m'

    class FG:
        BLACK = '\033[30m'
        RED = '\033[31m'
        GREEN = '\033[32m'
        ORANGE = '\033[33m'
        BLUE = '\033[34m'
        PURPLE = '\033[35m'
        CYAN = '\033[36m'
        LIGHT_GREY = '\033[37m'
        DARKGREY = '\033[90m'
        LIGHT_RED = '\033[91m'
        LIGHT_GREEN = '\033[92m'
        YELLOW = '\033[93m'         # yellow = "\033[0;33;21m"
        LIGHT_BLUE = '\033[94m'
        PINK = '\033[95m'
        LIGHT_CYAN = '\033[96m'

    class BG:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        ORANGE = '\033[43m'
        BLUE = '\033[44m'
        PURPLE = '\033[45m'
        CYAN = '\033[46m'
        LIGHT_GREY = '\033[47m'

    @staticmethod
    def print_color(color, text):
        print(color + text + ConsoleColors.RESET)

    @staticmethod
    def print_green(text):
        print(ConsoleColors.FG.GREEN + text + ConsoleColors.RESET)

    @staticmethod
    def print_error(text):
        print(ConsoleColors.FG.RED + text + ConsoleColors.RESET)


class CustomLoggingFormatter(logging.Formatter):
    """
    A logging formatter to automatically color errors, warnings, info and debug output text
    """

    FORMATS = {
        logging.ERROR: ConsoleColors.FG.RED + "ERROR: %(filename)s: %(message)s" + ConsoleColors.RESET,
        logging.WARNING: ConsoleColors.FG.RED + "WARNING: %(message)s" + ConsoleColors.RESET,
        logging.INFO: "%(message)s",                     # TODO: currently no explicit color set, breaks 'print_progressbar'
        logging.DEBUG: "DEBUG: %(message)s",
        "DEFAULT": "%(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def multi_line_list(input_array, indent_length=45, first_line_caption='', items_per_line=4, item_delimiter=' ', first_line_single_item=None, no_item_text='-'):
    """
    Convert an array of strings (list or tuple) to a formatted string where the items are
    equally spread over multiple lines, instead of one long line

    Args:
        input_array (arroy of str):    String to format as text
        indent_length (int):           The spacing in front of the listed items
        first_line_caption (str):      The text in the first line that precedes the listing of the items in the array.
        items_per_line (int):          The number of items per line
        item_delimiter (str):          The str that is added between the array-items in the text
        first_line_single_item (str):  If set, will display this string on the first line, the rest of the items will
                                       then be shown from the second line on
        no_item_text (str):            Text if there are no items

    """
    current_line = ''
    return_text = ''

    if len(input_array) == 0:
        return first_line_caption.ljust(indent_length, ' ') + no_item_text

    #
    if first_line_single_item is not None:
        return_text = first_line_caption.ljust(indent_length, ' ') + str(first_line_single_item)
    else:
        current_line = first_line_caption

    # generate and print the lines
    sub_line = ''
    for i in range(len(input_array)):
        if not len(sub_line) == 0:
            sub_line += item_delimiter
        sub_line += str(input_array[i])
        if (i + 1) % items_per_line == 0:
            if not len(return_text) == 0:
                return_text += '\n'
            return_text += current_line.ljust(indent_length, ' ') + sub_line
            sub_line = ''
            current_line = ''

    # print the remaining items (if there are any)
    if not len(sub_line) == 0:
        if not len(return_text) == 0:
            return_text += '\n'
        return_text += current_line.ljust(indent_length, ' ') + sub_line

    return return_text


def print_progressbar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar

    Args:
        iteration (int):    current iteration (Int)
        total (int):        total iterations (Int)
        prefix (str):       prefix string (Str)
        suffix (str)        suffix string (Str)
        decimals (int)      positive number of decimals in percent complete (Int)
        length (int):       character length of bar (Int)
        fill (str):         bar fill character (Str)
        print_end (str):    end character (e.g. "\r", "\r\n") (Str)
        color (str):        text color (as ASCII code)

        Source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=print_end)
    if iteration == total:
        print()

